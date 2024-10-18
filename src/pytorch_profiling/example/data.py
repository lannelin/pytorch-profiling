from collections.abc import Callable

import lightning as L
from torch.utils.data import DataLoader
from torchvision.datasets import Flowers102


class Flowers102DataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        train_transform: Callable | None = None,
        val_transform: Callable | None = None,
        test_transform: Callable | None = None,
        target_transform: Callable | None = None,
        num_workers: int = 0,
        persistent_workers: bool = False,
        download: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.target_transform = target_transform
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.download = download

    def setup(self, stage: str):

        if stage == "test" or stage is None:
            self.ds_test = Flowers102(
                root=self.data_dir,
                split="test",
                download=self.download,
                transform=self.test_transform,
                target_transform=self.target_transform,
            )

        if stage == "fit" or stage is None:
            # don't pass transform at this stage but on subset later

            self.ds_train = Flowers102(
                root=self.data_dir,
                split="train",
                download=self.download,
                transform=self.train_transform,
                target_transform=self.target_transform,
            )
            self.ds_val = Flowers102(
                root=self.data_dir,
                split="val",
                download=self.download,
                transform=self.val_transform,
                target_transform=self.target_transform,
            )

    def train_dataloader(self):
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.ds_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )
