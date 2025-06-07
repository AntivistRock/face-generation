from omegaconf import DictConfig


def create_model(cfg: DictConfig):
    if cfg["model"]["model_type"] == "vae":
        from .vae import VAE

        return VAE(cfg)
