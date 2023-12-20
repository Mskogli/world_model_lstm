import os
import torch
import argparse
import datetime
import numpy as np

from torch.utils.data import DataLoader

from GMMRNN import GMMRNN, detach
from dataset import AerialGymTrajDataset, split_dataset


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Script for training a model with custom settings."
    )

    # Add arguments with default values
    parser.add_argument(
        "--batch_size", type=int, default=200, help="Batch size for training"
    )
    parser.add_argument(
        "--seq_length", type=int, default=92, help="Length of input sequences"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=200, help="Number of training epochs"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Training device, e.g. cuda:0"
    )
    parser.add_argument("--traj_length", type=int, default=97, help="Trajectory length")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    run_id = datetime.datetime.now().time()
    run_path = f"{os.path.dirname(os.path.abspath(__file__))}/runs/{run_id}"
    os.mkdir(run_path)

    args = parse_arguments()
    batch_size, seq_length, num_epochs, device, traj_length = (
        args.batch_size,
        args.seq_length,
        args.num_epochs,
        torch.device(args.device),
        args.traj_length,
    )

    dataset = AerialGymTrajDataset(
        "/home/mathias/Dokumenter/dev_2/datasets/trajectories.jsonl",
        device,
        actions=True,
    )

    train_dataset, val_dataset = split_dataset(dataset, 0.1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    model = GMMRNN(
        input_dim=132, latent_dim=128, hidden_dim=2048, n_gaussians=1, device=device
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=2)

    min_val_loss = torch.inf

    prediction_length = 1
    seq_length = traj_length - prediction_length
    val_losses = []
    for epoch in range(num_epochs):
        for batch in train_loader:
            hidden = model.init_hidden_state(batch.size(0))

            inputs = batch[:, 0 : seq_length, :]
            targets = batch[
                :,
                (prediction_length) : traj_length,
                :128,
            ]

            # Forward pass
            hidden = detach(hidden)  # Truncated backprop trough time
            (logpi, mu, sigma), hidden = model(
                inputs[:, :, :128], inputs[:, :, 128:], hidden
            )
            loss = model.loss_criterion(targets, logpi, mu, sigma)

            model.zero_grad()
            loss.backward()
            optimizer.step()

        print(
            "Epoch [{}/{}], train loss: {:.4f}".format(
                epoch, num_epochs, loss.item()
            )
        )

        total_val_loss = 0
        model.eval()
        with torch.no_grad():
            for val_batch in val_loader:
                val_hidden = model.init_hidden_state(val_batch.size(0))

                val_inputs = val_batch[:, 0 : seq_length, :]
                val_targets = val_batch[
                    :,
                    (prediction_length) : (prediction_length) + seq_length,
                    :128,
                ]

                val_hidden = detach(val_hidden)
                (val_pi, val_mu, val_sigma), val_hidden = model(
                    val_inputs[:, :, :128], val_inputs[:, :, 128:], val_hidden
                )
                val_loss = model.loss_criterion(
                    val_targets, val_pi, val_mu, val_sigma
                )

                total_val_loss += val_loss.item()

            print(
                "Epoch [{}/{}], validation loss: {:.4f}".format(
                    epoch, num_epochs, total_val_loss
                )
            )
        val_losses.append(total_val_loss)
        if total_val_loss < min_val_loss:
            min_val_loss = total_val_loss
            torch.save(
                model.state_dict(), f"{run_path}/model_{epoch}_{total_val_loss}.pth"
            )

        model.train()
    
    with open("2048-200.npy", "wb") as f:
        np.save(f, np.array(val_losses))