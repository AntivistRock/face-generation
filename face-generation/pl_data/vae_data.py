import pytorch_lightning as pl
from dvc.repo import Repo
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CelebA


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
        print("DATA PATH:", self.cfg["data"]["path"])
        self.train_dataset = CelebA(
            self.cfg["data"]["path"],
            transform=self.celeb_transform,
            download=False,
            split="train",
        )
        self.val_dataset = CelebA(
            self.cfg["data"]["path"],
            transform=self.celeb_transform,
            download=False,
            split="valid",
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.cfg["model"]["batch_size"], shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.cfg["model"]["batch_size"], shuffle=True
        )
