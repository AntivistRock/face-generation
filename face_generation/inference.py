from face_generation.pl_models import create_model
from face_generation.pl_models.vae import VAE
from pathlib import Path
import torch
from omegaconf import DictConfig
import hydra
from torchvision.utils import save_image


def load_model(logs_path: Path, cfg: DictConfig):
    ckpts_path = logs_path / "checkpoints"
    ckpt = list(ckpts_path.iterdir())[0]
    pl_model = VAE.load_from_checkpoint(ckpt, cfg=cfg)
    return pl_model.model


@hydra.main('../configs', 'main', version_base="1.3")
def inference(cfg: DictConfig):
    logs_path = Path(__file__).parent.parent / "plots"
    model = load_model(logs_path, cfg)

    z = torch.randn(cfg["model"]["latent_dim"]).cuda()
    img = model.decode(z)


    img = img.view(3, cfg["model"]["image_size"], cfg["model"]["image_size"])
    save_image(img, logs_path / "gen_example.png", normalize=True)


if __name__ == "__main__":
    inference()