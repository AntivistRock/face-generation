from omegaconf import DictConfig


def create_data(cfg: DictConfig):
    if cfg["model"]["model_type"] == "vae":
        from .vae_data import VAEDataModule

        return VAEDataModule(cfg)
