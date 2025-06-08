import pytorch_lightning as pl
from dvc.repo import Repo
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import os
from pathlib import Path


class VAEDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.celeb_transform = transforms.Compose(
            [
                transforms.Resize(cfg["model"]["image_size"], antialias=True),
                transforms.CenterCrop(cfg["model"]["image_size"]),
                transforms.ToTensor(),
            ]
        )  # used when transforming image to tensor

    def prepare_data(self):
        return
        repo = Repo(".")
        repo.pull()

    def setup(self, stage):
        repo_path = Path(os.getcwd()).parent
        self.train_dataset = ImageFolder(
            repo_path / self.cfg["data"]["train_path"],
            transform=self.celeb_transform,
        )
        self.val_dataset = ImageFolder(
            repo_path / self.cfg["data"]["val_path"],
            transform=self.celeb_transform,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.cfg["model"]["batch_size"], shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.cfg["model"]["batch_size"], shuffle=True
        )
