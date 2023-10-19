import torch
import ujson

from torch.utils.data import Dataset, random_split


class AerialGymTrajDataset(Dataset):
    def __init__(
        self,
        json_path: str,
        device: str,
        validation_split: float = 0.1,
    ) -> None:
        with open(json_path) as file:
            print("===LOADING DATASET...===")
            self.lines = file.readlines()
            print("===DATASET LOADED!===")

        self.device = device

        # Compute the train and val splits
        self.validation_split = validation_split
        self.total_samples = len(self.lines)
        self.train_len = int(self.total_samples * (1 - validation_split))
        self.val_len = self.total_samples - self.train_len

        self.dataset = self.split_dataset()

    def __len__(self) -> int:
        return len(self.lines)

    def split_dataset(self):
        train_data, val_data = random_split(self, [self.train_len, self.val_len])
        return train_data, val_data

    def __getitem__(self, idx) -> torch.tensor:
        json_object = ujson.loads(self.lines[self.dataset.indices[idx]])
        latents = torch.tensor(json_object["latents"], device=self.device)
        states = torch.tensor(json_object["states"], device=self.device)
        actions = torch.tensor(json_object["action"], device=self.device)

        item = torch.cat((latents, states, actions), 1)

        return torch.tensor(item, device=self.device)
