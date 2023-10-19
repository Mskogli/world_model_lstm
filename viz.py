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
        states=["full state"],
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    seVAE = VAENetworkInterface(batch_size=2)

    model = GMMRNN(input_dim=148, latent_dim=128, hidden_dim=1024, n_gaussians=2).to(
        device
    )
    model.load_state_dict(
        torch.load("/home/mathias/dev/world_model_lstm/baselines/model_2.pth")
    )
    model.eval()

    data = next(iter(dataloader))
    hidden = model.init_hidden_state(1)

    zero = np.random.randint(data.size(0))
    one = np.random.randint(data.size(1))

    x = data[zero : zero + 1, 0:32, :]
    y = data[zero : zero + 1, 32:33, :]

    (logpi, mu, sigma), hidden = model(x, hidden)

    pi = torch.exp(logpi[:, -1, :]).view(-1)
    mu = mu[:, -1, :, :].view(2, 128)
    sigma = sigma[:, -1, :, :].view(2, 128)

    pred_next_latent = sample_gmm(pi, mu, sigma)
    # pred_next_latent = pi[0] * mu[0] + pi[1] * mu[1]

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
