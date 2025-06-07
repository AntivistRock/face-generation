import pytorch_lightning as pl
import torch.nn as nn
import torch.functional as F
import torch
from scratch_models.vae import VAE as model_VAE
from omegaconf import DictConfig


class VAE(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.model = model_VAE()
        self.cfg = cfg
    
    def vae_loss(self, recon_x, x, mu, log_var):
            MSE = F.mse_loss(recon_x, x.view(-1, self.cfg['image_dim']))
            KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
            kld_weight = 0.00025
            loss = MSE + kld_weight * KLD  
            return loss
    
    def forward(self, x):
        out = self.model(x)
        return out

    def configure_optimizers(self):
        optimizer = nn.optim.Adam(self.parameters, lr=1e-3)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        torch.cuda.empty_cache()
        data = train_batch
        recon_batch, mu, log_var = self.model(data)
        log_var = torch.clamp_(log_var, -10, 10)
        loss = self.vae_loss(recon_batch, data, mu, log_var)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        data = val_batch
        recon_batch, mu, log_var = self.model(data)
        test_loss += self.vae_loss(recon_batch, data, mu, log_var).item()
        if batch_idx == 0:
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n],
                                    recon_batch.view(self.cfg['batch_size'], 3, self.cfg['image_size'], self.cfg['image_size'])[:n]])
            self.logger.experiment.log_image(comparison.cpu(), key='vae_gen')
