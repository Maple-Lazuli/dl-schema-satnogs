"""
Torch Datasets serve to retrieve features and labels one sample at a time. Accordingly,
the dataset must implement __len__ and __getitem__.

Torch Dataloaders are iterables that abstract batching, shuffling, and multiprocessing.
"""
from pathlib import Path

import torch
from PIL import Image
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms

from dl_schema.cfg import TrainConfig


class MNISTDataset(Dataset):
    """Sample torch Dataset to be used with torch DataLoader."""

    def __init__(
            self,
            split="train",
            cfg=TrainConfig(),
    ):
        assert split in {"train", "test"}
        self.cfg = cfg
        self.root = Path(getattr(self.cfg.data, f"{split}_root")).expanduser()
        self.img_root = self.root / "images"
        label_root = self.root / "labels"
        # load appropriate labels csv as dataframe
        self.labels = pd.read_csv(
            label_root / "annot.csv", header=None, names=["filename", "digit_id"]
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        filename = self.labels["filename"][idx]
        digit_id = self.labels["digit_id"][idx]
        img_path = self.img_root / filename
        # load image as np.uint8 shape (28, 28)
        x = Image.open(img_path)
        x = np.array(x)
        # convert to [0, 1.0] torch.float32, and normalize
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        x = transform(x)

        return x, digit_id


class SatnogsDataset(Dataset):

    def __init__(self, csv):
        self.annotations = pd.read_csv(csv)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        example = self.annotations.iloc[index]
        img = np.fromfile(example['waterfall_location'], dtype=np.uint8).reshape(-1, 623)
        #img = Image.fromarray(img)
        #normalize psds
        #
        img = torch.from_numpy(img).type(torch.float)
        img = img.unsqueeze(0).repeat(3,1,1)
        target = torch.tensor(example['status']).unsqueeze(0).type(torch.float)
        return img, target


if __name__ == "__main__":
    """Test Dataset"""

    train_data = SatnogsDataset("/home/maple/CodeProjects/satnogs_cnn/data/train.csv")
    #test_data = SatnogsDataset
    print(train_data[0][1].dtype)
    #print(test_data[0])
