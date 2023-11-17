import torch
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader

from GMMRNN import GMMRNN
from gmm import sample_gmm
from dataset import AerialGymTrajDataset
from sevae.inference.scripts.VAENetworkInterface import VAENetworkInterface

if __name__ == "__main__":
    device = torch.device("cpu")

    dataset = AerialGymTrajDataset(
        "/Users/mathias/Documents/trajectories.jsonl",
        device,
        actions=True,
    )
    dataloader = DataLoader(dataset, batch_size=100, shuffle=False)

    seVAE = VAENetworkInterface(device=device)

    model = GMMRNN(
        input_dim=132, latent_dim=128, hidden_dim=512, n_gaussians=5, device=device
    ).to(device)
    model.load_state_dict(
        torch.load(
            "/Users/mathias/dev/world_model_lstm/runs/13:31:42.338124/model_9_1981.7920227050781.pth"
        )
    )
    model.eval()

    data = next(iter(dataloader))
    hidden = model.init_hidden_state(1)

    zero = np.random.randint(data.size(0))
    one = np.random.randint(data.size(1))

    x = data[zero : zero + 1, 0:82, :]
    y = data[zero : zero + 1, 86:87, :]
    z = data[zero : zero + 1, 81:82, :]

    (logpi, mu, sigma), hidden = model(x[:, :, :128], x[:, :, 128:], hidden)

    pi = torch.exp(logpi[:, -1, :]).view(-1)
    mu = mu[:, -1, :, :].view(5, 128)
    sigma = sigma[:, -1, :, :].view(5, 128)

    pred_next_latent = sample_gmm(pi, mu, sigma)

    prev_hidden = hidden
    future_actions = torch.tensor(
        [
            -0.11876555532217026,
            -0.07435929775238037,
            -0.5583888292312622,
            0.040141552686691284,
        ],
        device=torch.device("cuda:0"),
    ).view(1, 1, 4)
    next_input = torch.cat((pred_next_latent.view(1, 1, -1), future_actions), -1)
    fig = plt.figure(figsize=(3, 3))
    fig = plt.figure(figsize=(20, 10), dpi=100, facecolor="w", edgecolor="k")
    for i in range(20):
        fig.add_subplot(2, 10, i + 1)
        (logpi, mu, sigma), hidden = model(next_input, prev_hidden)
        pi = torch.exp(logpi)
        next_latent = sample_gmm(pi, mu, sigma)
        next_input = torch.cat((next_latent.view(1, 1, -1), future_actions), -1)
        pred_depth = seVAE.decode(next_latent.view(1, -1))
        prev_hidden = hidden
        plt.tick_params(
            left=False, right=False, labelleft=False, labelbottom=False, bottom=False
        )
        plt.imshow(pred_depth.view(270, 480).cpu().detach().numpy())
    plt.show()

    gt_depth_img = seVAE.decode(y[:, :, :128].view(1, -1))
    gt_depth_img_2 = seVAE.decode(z[:, :, :128].view(1, -1))
    pred_depth_img = seVAE.decode(pred_next_latent.view(1, -1))

    print(pred_depth_img)

    fig = plt.figure(figsize=(1, 3))
    fig = plt.figure(figsize=(12, 8), dpi=100, facecolor="w", edgecolor="k")
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_title("Depth image at time t")
    plt.imshow(gt_depth_img_2.reshape(270, 480))
    ax3 = fig.add_subplot(1, 3, 2)
    ax3.set_title("GT depth image predition at time t+5")
    plt.imshow(gt_depth_img.reshape(270, 480))
    ax2 = fig.add_subplot(1, 3, 3)
    ax2.set_title("Predicted depth image at time t+5")
    plt.imshow(pred_depth_img.reshape(270, 480))
    plt.show()

# %%
