import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from GMMRNN import GMMRNN
from gmm import sample_gmm
from dataset import AerialGymTrajDataset
from sevae.inference.scripts.VAENetworkInterface import VAENetworkInterface

if __name__ == "__main__":
    context_length = 95
    prediction_length = 1
    dream_horizon = 5

    device = torch.device("cuda:0")

    dataset = AerialGymTrajDataset(
        "/home/mathias/Dokumenter/dev_2/datasets/trajectories.jsonl",
        device,
        actions=True,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    seVAE = VAENetworkInterface()
    model = GMMRNN(input_dim=132, latent_dim=128, hidden_dim=2048, n_gaussians=10).to(
        device
    )

    model.load_state_dict(
        torch.load("/home/mathias/Dokumenter/dev_2/world_model_lstm/runs/10:30:36.557068/model_479_2383.516860961914.pth")
    )
    model.eval()

    data = next(iter(dataloader))
    hidden = model.init_hidden_state(1)

    traj_context = data[:, 0:context_length, :]

    #gt_dream = data[:, context_length+1:context_length+1+dream_horizon, :]

    (logpi, mu, sigma), hidden = model(traj_context[..., :model.latent_dim], traj_context[..., model.latent_dim:], hidden)

    pi = torch.exp(logpi[:, -1, :]).view(-1)
    mu = mu[:, -1, :, :].view(model.n_gaussians, model.latent_dim)
    sigma = sigma[:, -1, :, :].view(model.n_gaussians, model.latent_dim)
    pred_next_latent, _ = sample_gmm(pi, mu, sigma)

    current_latent = data[:, context_length - 1:context_length, :]
    gt_next_latent = data[:, context_length:context_length + prediction_length, :]
    #dreamed_latent, _ = model.dream(dream_horizon, hidden, gt_next_latent[:, :, :model.latent_dim], gt_dream)

    current_depth_img = seVAE.decode(current_latent[:, :, :model.latent_dim].view(1, -1)) 
    gt_depth_img = seVAE.decode(gt_next_latent[:, :, :model.latent_dim].view(1, -1))
    pred_depth_img = seVAE.decode(pred_next_latent.view(1, -1))
    #dreamed_depth_img = seVAE.decode(dreamed_latent.view(1, -1))


    fig = plt.figure(figsize=(12, 4), dpi=100, facecolor="w", edgecolor="k")

    ax1 = fig.add_subplot(1,3,1)
    ax1.set_title("$l_t$")
    plt.imshow(current_depth_img.reshape(270, 480))
    plt.axis("off")

    ax2 = fig.add_subplot(1,3,2)
    ax2.set_title("$\hat l_{t+1}$")
    plt.imshow(pred_depth_img.reshape(270, 480))
    plt.axis("off")

    ax3 = fig.add_subplot(1,3,3)
    ax3.set_title("$l_{t+1}$")
    plt.imshow(gt_depth_img.reshape(270, 480))
    plt.axis("off")

    fig2 = plt.figure(figsize=(23, 5.5), dpi=100, facecolor="w", edgecolor="k")

    for i in range(model.n_gaussians):
        print(pi[i].round(decimals=1).item())
        model_pred_depth_img = seVAE.decode(mu[i, :].view(1, -1))
        ax = fig2.add_subplot(2, int(model.n_gaussians/2), i+1)
        ax.set_title(f"$p ={round(pi[i].item(), 3)}$", fontsize=18)
        plt.imshow(model_pred_depth_img.reshape(270, 480))
        plt.axis("off")

    plt.show()
    

# %%
