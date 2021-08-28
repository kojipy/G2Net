import os
import sys

import pandas as pd
import numpy as np
import torch
from nnAudio.Spectrogram import CQT1992v2
from torch.utils.data import Dataset

sys.path.append(os.path.abspath(f"{__file__}/../../../"))

from src.config import Config


class TrainDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        print("========== Set Up Dataset ==========")
        self.df = df
        self.file_names = df["file_path"].values
        self.labels = df[Config.target_col].values
        self.wave_transform = CQT1992v2(sr=2048, fmin=20, fmax=1024, hop_length=64)
        self.transform = transform
        print("========== Completed Dataset Setup ==========")

    def __len__(self):
        return len(self.df)

    def apply_qtransform(self, waves, transform):
        waves = np.hstack(waves)
        waves = waves / np.max(waves)
        waves = torch.from_numpy(waves).float()
        image = transform(waves)
        return image

    def __getitem__(self, idx):
        file_path = self.file_names[idx]
        waves = np.load(file_path)
        image = self.apply_qtransform(waves, self.wave_transform)
        if self.transform:
            image = image.squeeze().numpy()
            image = self.transform(image=image)["image"]
        label = torch.tensor(self.labels[idx]).float()
        return image, label
