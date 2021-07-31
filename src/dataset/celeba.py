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
from glob import glob


class CelebADataset(torch.utils.data.Dataset):
    def __init__(self,
            root: str = '/src/datasets/celeba',
            is_train: bool = False,
            transform=None
        ) -> None:
        super().__init__()

        self.transform = transform

        self.files = glob(os.path.join(root, '*.png'))
        
        ids = list(range(len(self.files)))
        if is_train:
            self.ids = ids[0:int(len(ids)*0.8)]
        else:
            self.ids = ids[int(len(ids)*0.8):]

    def __len__(self) -> None:
        return len(self.ids)

    def __getitem__(self, key: Union[int, slice]) -> Union[Tensor, List[Tensor]]:
        img, label = self.dataset[key]
        
        img = img.convert('RGB')
        img = self.transform(img) # [3, 28, 28], range: [-1, 1]
        
        label = F.one_hot(torch.tensor(label), num_classes=10) # [10]

        return img, label


class TrainDataset(CelebADataset):
    def __init__(self) -> None:
        transform = T.Compose([
            T.Resize((256, 256)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=0.5, std=0.5),
        ])
        super().__init__(is_train=True, transform=transform)
        

class TestDataset(CelebADataset):
    def __init__(self) -> None:
        transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(mean=0.5, std=0.5),
        ])

        super().__init__(is_train=False, transform=transform)


if __name__ == '__main__':
    d = TrainDataset() # 60,000
    d1 = TestDataset() # 10,000
    print(len(d), len(d1))