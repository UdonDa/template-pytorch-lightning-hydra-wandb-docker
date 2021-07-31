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
from torchvision.datasets import MNIST


class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self,
            root: str = '/src/datasets/mnist',
            is_train: bool = False
        ) -> None:
        super().__init__()

        self.dataset = MNIST(root, train=is_train, download=True)


    def __len__(self) -> None:
        return len(self.dataset)

    def __getitem__(self, key: Union[int, slice]) -> Union[Tensor, List[Tensor]]:
        return self.dataset[key]


class TrainDataset(MNISTDataset):
    def __init__(self) -> None:
        super().__init__()
        # like a dataset of images
        # self.dataset = MNISTDataset(is_train=True)
        print('a')


class ValDataset(MNISTDataset):
    def __init__(self) -> None:
        super().__init__()
        self.dataset = [torch.randn(1, 28, 28) for _ in range(10000)]


class TestDataset(MNISTDataset):
    def __init__(self) -> None:
        super().__init__()
        self.dataset = [torch.randn(1, 28, 28) for _ in range(10000)]


if __name__ == '__main__':
    d = TrainDataset()
    print('a')