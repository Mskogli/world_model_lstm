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
    context_length = 40
    prediction_length = 1

    device = torch.device("cuda:0")
    dataset = AerialGymTrajDataset(
        "/home/mathias/Dokumenter/dev_2/aerial_gym_simulator/aerial_gym/rl_training/rl_games/static_object_exp.jsonl",
        device,
        actions=True,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    seVAE = VAENetworkInterface()
    model = GMMRNN(input_dim=132, latent_dim=128, hidden_dim=1024, n_gaussians=10).to(
        device
    )

    model.load_state_dict(
        torch.load("/home/mathias/Dokumenter/dev_2/world_model_lstm/runs/10_gaussians_pred_1/model_130_2549.6763610839844.pth")
    )
    model.eval()

    data = next(iter(dataloader))
    hidden = model.init_hidden_state(1)

    zero = np.random.randint(data.size(0))

    traj_context = data[:, 0:context_length, :]


    (logpi, mu, sigma), hidden = model(traj_context[..., :model.latent_dim], traj_context[..., model.latent_dim:], hidden)

    pi = torch.exp(logpi[:, -1, :]).view(-1)
    mu = mu[:, -1, :, :].view(model.n_gaussians, model.latent_dim)
    sigma = sigma[:, -1, :, :].view(model.n_gaussians, model.latent_dim)
    pred_next_latent = sample_gmm(pi, mu, sigma)

    #dreamed_latent, _ = model.dream(5, hidden, mu)

    current_latent = data[:, context_length - 1:context_length, :]
    gt_next_latent = data[:, context_length:context_length + prediction_length, :]


    current_depth_img = seVAE.decode(current_latent[:, :, :model.latent_dim].view(1, -1)) 
    gt_depth_img = seVAE.decode(gt_next_latent[:, :, :model.latent_dim].view(1, -1))
    pred_depth_img = seVAE.decode(pred_next_latent.view(1, -1))
    #dreamed_depth_img = seVAE.decode(dreamed_latent.view(1, -1))


    fig = plt.figure(figsize=(12, 4), dpi=100, facecolor="w", edgecolor="k")

    ax1 = fig.add_subplot(1,3,1)
    ax1.set_title("$l_t$")
    ax1.axhline(y=113, color='r', linewidth=0.5)
    plt.imshow(current_depth_img.reshape(270, 480))
    plt.axis("off")

    ax2 = fig.add_subplot(1,3,2)
    ax2.set_title("$\hat l_{t+1}$")
    plt.imshow(pred_depth_img.reshape(270, 480))
    ax2.axhline(y=113, color='r', linewidth=0.5)
    plt.axis("off")

    ax3 = fig.add_subplot(1,3,3)
    ax3.set_title("$l_{t+1}$")
    plt.imshow(gt_depth_img.reshape(270, 480))
    ax3.axhline(y=113, color='r', linewidth=0.5)
    plt.axis("off")

 
    fig2 = plt.figure(figsize=(30, 5), dpi=100, facecolor="w", edgecolor="k")

    for i in range(model.n_gaussians):
        print(pi[i].round(decimals=1).item())
        model_pred_depth_img = seVAE.decode(mu[i, :].view(1, -1))
        ax = fig2.add_subplot(2, int(model.n_gaussians/2), i+1)
        ax.set_title(f"$p ={round(pi[i].item(), 3)}$")
        plt.imshow(model_pred_depth_img.reshape(270, 480))
        ax.axhline(y=135, color='r', linewidth=0.5)
        plt.axis("off")


    plt.show()
