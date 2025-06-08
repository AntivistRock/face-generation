import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from omegaconf import DictConfig
from torchvision import transforms

from face_generation.scratch_models.vae import VAE as model_VAE


class VAE(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.model = model_VAE(cfg)
        self.cfg = cfg["model"]

    def vae_loss(self, recon_x, x, mu, log_var):
        MSE = F.mse_loss(recon_x, x.view(-1, self.cfg["image_dim"]))
        KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        kld_weight = 0.00025
        loss = MSE + kld_weight * KLD
        return loss

    def forward(self, x):
        out = self.model(x)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        torch.cuda.empty_cache()
        data, _ = train_batch
        recon_batch, mu, log_var = self.model(data)
        log_var = torch.clamp_(log_var, -10, 10)
        loss = self.vae_loss(recon_batch, data, mu, log_var)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        data, _ = val_batch
        recon_batch, mu, log_var = self.model(data)
        loss = self.vae_loss(recon_batch, data, mu, log_var).item()
        self.log("test_loss", loss, on_epoch=True)
        if batch_idx == 0:
            n = min(data.size(0), 4)
            comparison = torch.cat(
                [
                    data[:n],
                    recon_batch.view(
                        self.cfg["batch_size"],
                        3,
                        self.cfg["image_size"],
                        self.cfg["image_size"],
                    )[:n],
                ]
            )
            grid = vutils.make_grid(comparison, nrow=n, padding=2, normalize=True)
            comparison = transforms.ToPILImage()(grid)
            self.logger.experiment.log_image(
                run_id=self.logger.run_id,
                image=comparison,
                key="vae_gen",
                step=self.current_epoch,
            )
