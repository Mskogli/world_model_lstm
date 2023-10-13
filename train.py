import torch
import jsonlines

from torch.utils.data import Dataset, DataLoader
from LSTMWordlEncoder import LSTMWorldModel, KLregularizedLogLikelihoodLoss, detach

# TODO: Add argument parsing to set training-hyper params

class TrajectoryDataset(Dataset):
    pass


device = torch.device("cuda:0")
batch_size = 100
num_epochs = 500
sequence_length = 16
trajectory_length = 150


data = torch.zeros((182, 297, 128), device=device)
reader = jsonlines.open('/home/mathias/dev/world_models/data/trajectories.jsonl')
i = 0
for obj in reader:
    print(reader.next())
    if len(obj["latents"]) == 297:
        print(i)
        data[i, :, :]  = torch.tensor((obj["latents"]), device=device)
        i += 1


data = data.view(-1, 297, 128)


if __name__ == "__main__":
    model = LSTMWorldModel(latent_dim=128, hidden_dim=256).to(device)
    optimizer = torch.optim.Adam(model.parameters())

    hidden = (torch.zeros(1, 182, 256).to(device),
             torch.zeros(1, 182, 256).to(device))

    for epoch in range(1000):

        for i in range(0, trajectory_length - sequence_length, sequence_length):
            inputs = data[:, i:i + sequence_length, :]
            targets = data[:, (i + 1):(i + 1) + sequence_length, :]

            # Forward pass
            hidden = detach(hidden) # Truncated BBPT   
            (mu, sigma), hidden = model(inputs, hidden)
            loss = KLregularizedLogLikelihoodLoss(targets, mu, sigma)

            model.zero_grad()
            loss.backward()
            optimizer.step()

            print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch, num_epochs, loss.item()))
