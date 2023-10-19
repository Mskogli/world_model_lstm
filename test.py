import torch
import sevae.inference.scripts.VAENetworkInterface as seVAE
import matplotlib.pyplot as plt
import numpy as np

from LSTMWordlEncoder import LSTMWorldModel
from train import AerialGymTrajDataset

from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda:0")
dataset = AerialGymTrajDataset('/home/mathias/dev/aerial_gym_simulator/aerial_gym/rl_training/rl_games/trajectories.jsonl', device)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True) 
semantically_enhanced_autoencoder = seVAE.VAENetworkInterface(batch_size=2)
model = LSTMWorldModel(input_dim=148, latent_dim=128, hidden_dim=1024, n_gaussians=2).to(device)


def sample_from_gmm(weights: torch.tensor, mu: torch.tensor, sigma: torch.tensor, num_samples: int = 1) -> torch.tensor:
    """
    Draw samples from a gaussian mixture model
    """
    pass


if __name__ == "__main__":
    model.load_state_dict(torch.load("/home/mathias/dev/world_model_lstm/model.pth"))
    model.eval()
    data = next(iter(dataloader))
    hidden = (torch.zeros(1, 1, 1024).to(device),
            torch.zeros(1, 1, 1024).to(device))
    
    zero = np.random.randint(data.size(0)) 
    one = np.random.randint(data.size(1))

    x = data[zero:zero+1, 0:1, :]
    y = data[zero:zero+1, 1:2, :]

    (logpi, mu, sigma), hidden = model(x, hidden)

    pi = torch.exp(logpi[:, -1, :] ).view(-1)
    mu = mu[:, -1, :, :].view(2, 128) 
    sigma = sigma[:, -1, :, :].view(2, 128)

    next_latent_mean = pi[0]*mu[0] + pi[1]*mu[1]
    print(next_latent_mean)
    print(y)

    y_depth_img = semantically_enhanced_autoencoder.decode(y[:, :, :128].view(1, -1))
    x_depth_img = semantically_enhanced_autoencoder.decode(next_latent_mean.view(1, -1))

    print(pi)

    fig = plt.figure(figsize=(1, 2))
    fig.add_subplot(1, 2, 1)
    plt.imshow(y_depth_img.view(270, 480).cpu().detach().numpy())
    fig.add_subplot(1, 2, 2)
    plt.imshow(x_depth_img.view(270, 480).cpu().detach().numpy())
    plt.show()