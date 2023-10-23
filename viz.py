# %%
import torch
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader

from GMMRNN import GMMRNN
from gmm import sample_gmm
from dataset import AerialGymTrajDataset
from sevae.inference.scripts.VAENetworkInterface import VAENetworkInterface

if __name__ == "__main__":
    device = torch.device("cuda:0")

    dataset = AerialGymTrajDataset(
        "/home/mathias/dev/aerial_gym_simulator/aerial_gym/rl_training/rl_games/trajectories.jsonl",
        device,
        actions=True,
    )
    dataloader = DataLoader(dataset, batch_size=100, shuffle=False)

    seVAE = VAENetworkInterface(batch_size=1)

    model = GMMRNN(input_dim=132, latent_dim=128, hidden_dim=1024, n_gaussians=10).to(
        device
    )
    model.load_state_dict(
        torch.load(
            "/home/mathias/dev/world_model_lstm/runs/20:36:36.466796/model_146_1711.3021907806396.pth"
        )
    )
    model.eval()

    data = next(iter(dataloader))
    hidden = model.init_hidden_state(1)

    zero = np.random.randint(data.size(0))
    one = np.random.randint(data.size(1))

    batch = 1
    x = data[batch-1:batch, 100:115, :]
    y = data[batch-1:batch, 115:116, :]

    (logpi, mu, sigma), hidden = model(x, hidden)
    print(hidden[1].size())

    pi = torch.exp(logpi[:, -1, :]).view(-1)
    mu = mu[:, -1, :, :].view(10, 128)
    sigma = sigma[:, -1, :, :].view(10, 128)

    pred_next_latent = sample_gmm(pi, mu, sigma)
    # pred_next_latent = pi[0] * mu[0] + pi[1] * mu[1]

    prev_hidden = hidden
    future_actions = torch.tensor(
        [-0.11876555532217026, -0.07435929775238037, -0.5583888292312622, 0.040141552686691284], device=torch.device("cuda:0")
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
    pred_depth_img = seVAE.decode(pred_next_latent.view(1, -1))

    fig = plt.figure(figsize=(1, 2))
    fig = plt.figure(figsize=(12, 8), dpi=100, facecolor="w", edgecolor="k")
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title("GT Depth Image")
    plt.imshow(gt_depth_img.view(270, 480).cpu().detach().numpy())
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title("Pred Depth Image")
    plt.imshow(pred_depth_img.view(270, 480).cpu().detach().numpy())
    plt.show()

# %%
