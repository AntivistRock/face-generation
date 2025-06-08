import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.loggers import MLFlowLogger

from face_generation.pl_data import create_data
from face_generation.pl_models import create_model


@hydra.main("../configs", "main", version_base="1.3")
def train(cfg: DictConfig):
    mlf_logger = MLFlowLogger(
        experiment_name=cfg["model"]["model_type"],
        tracking_uri=cfg["logging"]["tracking_uri"],
    )
    model = create_model(cfg)
    dm = create_data(cfg)

    hyperparams = {
        "model type": cfg["model"]["model_type"],
        "latent dim": cfg["model"]["latent_dim"],
        "image size": cfg["model"]["image_size"],
        "epochs": cfg["model"]["epochs"],
        "batch size": cfg["model"]["batch_size"],
        "image_mean": cfg["model"]["image_mean"],
        "image_std": cfg["model"]["image_std"],
    }

    mlf_logger.log_hyperparams(hyperparams)

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath="./plots/checkpoints", save_top_k=1, monitor="test_loss"
        )
    ]

    trainer = pl.Trainer(
        accelerator="cuda",
        logger=mlf_logger,
        callbacks=callbacks,
        max_epochs=cfg["model"]["epochs"],
    )
    trainer.fit(model, dm)


if __name__ == "__main__":
    train()
