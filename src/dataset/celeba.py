from typing import List, Union

import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import Tensor
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST


class Dataset(torch.utils.data.Dataset):
    def __len__(self) -> None:
        return len(self.dataset)

    def __getitem__(self, key: Union[int, slice]) -> Union[Tensor, List[Tensor]]:
        return self.dataset[key]


class TrainDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        # like a dataset of images
        self.dataset = [torch.randn(1, 28, 28) for _ in range(80000)]


class ValDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.dataset = [torch.randn(1, 28, 28) for _ in range(10000)]


class TestDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.dataset = [torch.randn(1, 28, 28) for _ in range(10000)]
