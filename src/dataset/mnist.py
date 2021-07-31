from typing import List, Union

import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch import Tensor
from torchvision.datasets import MNIST


class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self,
            root: str = '/src/datasets/mnist',
            is_train: bool = False
        ) -> None:
        super().__init__()

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=0.5, std=0.5),
        ])

        self.dataset = MNIST(root, train=is_train, download=True)

    def __len__(self) -> None:
        return len(self.dataset)

    def __getitem__(self, key: Union[int, slice]) -> Union[Tensor, List[Tensor]]:
        img, label = self.dataset[key]
        
        img = img.convert('RGB')
        img = self.transform(img) # [3, 28, 28], range: [-1, 1]
        
        label = F.one_hot(torch.tensor(label), num_classes=10) # [10]

        return img, label


class TrainDataset(MNISTDataset):
    def __init__(self) -> None:
        super().__init__(is_train=True)
        

class TestDataset(MNISTDataset):
    def __init__(self) -> None:
        super().__init__(is_train=False)


if __name__ == '__main__':
    d = TrainDataset() # 60,000
    d1 = TestDataset() # 10,000
    print(len(d), len(d1))