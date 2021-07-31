import os
from typing import Optional

import pytorch_lightning as pl
from src.dataset.dataset import TestDataset, TrainDataset, ValDataset
from torch.utils.data import DataLoader
from hydra.utils import instantiate


class DataModule(pl.LightningDataModule):
    def __init__(self,
            config,
            ) -> None:
        super().__init__()

        self.train = None
        self.val = None
        self.test = None

        self.config = config

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        pass

    def train_dataloader(self) -> DataLoader:
        if self.config.train_dataloader._target_ is not None:
            return instantiate(self.config.train_dataloader)

    def val_dataloader(self) -> DataLoader:
        if self.config.val_dataloader._target_ is not None:
            return instantiate(self.config.val_dataloader)

    def test_dataloader(self) -> DataLoader:
        if self.config.test_dataloader._target_ is not None:
            return instantiate(self.config.test_dataloader)

    def teardown(self, stage: Optional[str] = None):
        # clean up after fit or test
        # called on every process in DDP
        pass
