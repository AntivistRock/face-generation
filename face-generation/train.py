import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pl_data import create_data
from pl_models import create_model
from pytorch_lightning.loggers import MLFlowLogger


@hydra.main("../configs", "main", version_base="1.3")
def train(cfg: DictConfig):
    mlf_logger = MLFlowLogger(
        experiment_name=cfg["model"]["model_type"],
        tracking_uri=cfg["logging"]["tracking_uri"],
    )
    model = create_model(cfg)
    dm = create_data(cfg)

    trainer = pl.Trainer(accelerator="cuda", devices=1, logger=mlf_logger)
    trainer.fit(model, dm)


if __name__ == "__main__":
    train()
