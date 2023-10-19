import torch
import ujson

from torch.utils.data import Dataset, random_split
from typing import Literal, List

States = Literal[
    "full state",
    "position",
    "attitude",
    "linear velocities",
    "angular velocities",
    "actions",
]

# The state is logged as 13 element vector in the dataset
StateIndices = {
    "full state": [0, 12],
    "position": [0, 3],
    "attitude": [3, 7],
    "linear velocities": [7, 10],
    "angular velocities": [10, 13],
}


class AerialGymTrajDataset(Dataset):
    def __init__(
        self,
        json_path: str,
        device: str,
        validation_split: float = 0.1,
        actions: bool = False,
        states: List[States] = None,
    ) -> None:
        print("===LOADING DATASET===")
        with open(json_path) as file:
            self.lines = file.readlines()
        print("===DATASET LOADED===")

        self.device = device

        self.actions = actions
        self.states = states

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

        latents = [torch.tensor(json_object["latents"], device=self.device)]
        actions = (
            [torch.tensor(json_object["action"], device=self.device)]
            if self.states
            else []
        )
        states = [
            torch.tensor(
                json_object["states"][StateIndices[state][0] : StateIndices[state][-1]],
                device=self.device,
            )
            for state in states
        ]

        item = torch.cat(latents + states + actions, 1)

        return item
