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
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

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
