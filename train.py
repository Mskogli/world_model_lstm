import torch
import ujson

from typing import Any
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from LSTMWordlEncoder import LSTMWorldModel, KLregularizedLogLikelihoodLoss, detach, criterion, gaussian_ll_loss

# TODO: Add argument parsing to set training-hyper params

class AerialGymTrajDataset(Dataset):
    def __init__(self, json_path: str, device: str) -> None:
        with open(json_path) as file:
            print("===LOADING DATASET...===")
            self.lines = file.readlines()
            print("===DATASET LOADED===")
            self.device = device

    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(self, idx) -> Any:
        json_object = ujson.loads(self.lines[idx])
        latents = torch.tensor(json_object["latents"], device=self.device) 
        states = torch.tensor(json_object["states"], device=self.device)
        actions = torch.tensor(json_object["action"], device=self.device)

        item = torch.cat((latents, states, actions), 1)

        if len(item) == 97:
            return torch.tensor(item, device=self.device)
        return torch.zeros((97, 132), device=self.device)

device = torch.device("cuda:0")
BATCH_SIZE = 500
NUM_EPOCHS = 10
SEQ_LENGTH = 32 # Sequence length
TRAJ_LENGTH = 97 # Trajectoy length

if __name__ == "__main__":
    dataset = AerialGymTrajDataset('/home/mathias/dev/aerial_gym_simulator/aerial_gym/rl_training/rl_games/trajectories.jsonl', device)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True) 
    model = LSTMWorldModel(input_dim=148, latent_dim=128, hidden_dim=1024, n_gaussians=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=10e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=2)


    for epoch in range(NUM_EPOCHS):

        for batch in dataloader:
            hidden = (torch.zeros(1, batch.size(0), 1024).to(device),
                torch.zeros(1, batch.size(0), 1024).to(device))

            for i in range(0, TRAJ_LENGTH - SEQ_LENGTH, SEQ_LENGTH):
                inputs = batch[:, i:i + SEQ_LENGTH, :]
                targets = batch[:, (i + 1):(i + 1) + SEQ_LENGTH, :128]

                # Forward pass
                hidden = detach(hidden)   
                (pi, mu, sigma), hidden = model(inputs, hidden)
                loss = criterion(targets, pi, mu, sigma)
                
                model.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()

            print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch, NUM_EPOCHS, loss.item()))

    torch.save(model.state_dict(), "model.pth")
