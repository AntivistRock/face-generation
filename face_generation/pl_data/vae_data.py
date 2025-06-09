import pytorch_lightning as pl
from dvc.repo import Repo
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


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
        repo = Repo(".")
        repo.pull(
            targets=[
                self.cfg["data_conf"]["train_path"],
                self.cfg["data_conf"]["val_path"],
            ],
            remote="data",
        )

    def setup(self, stage):
        self.train_dataset = ImageFolder(
            self.cfg["data_conf"]["train_path"],
            transform=self.celeb_transform,
        )
        self.val_dataset = ImageFolder(
            self.cfg["data_conf"]["val_path"],
            transform=self.celeb_transform,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.cfg["model"]["batch_size"], shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.cfg["model"]["batch_size"], shuffle=False
        )
