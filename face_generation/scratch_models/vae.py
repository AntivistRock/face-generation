import torch
from omegaconf import DictConfig
from torch import nn
from torchvision import transforms


class VAE(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(VAE, self).__init__()

        self.cfg = cfg
        hidden_dims = [32, 64, 128, 256, 512]
        self.final_dim = hidden_dims[-1]
        in_channels = 3
        modules = []

        self.celeb_transform1 = transforms.Compose(
            [
                transforms.Resize(self.cfg["model"]["image_size"], antialias=True),
                transforms.CenterCrop(self.cfg["model"]["image_size"]),
            ]
        )  # used by decode method to transform final output

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels=h_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        out = self.encoder(
            torch.rand(
                1, 3, self.cfg["model"]["image_size"], self.cfg["model"]["image_size"]
            )
        )
        self.size = out.shape[2]
        self.fc_mu = nn.Linear(
            hidden_dims[-1] * self.size * self.size, self.cfg["model"]["latent_dim"]
        )
        self.fc_var = nn.Linear(
            hidden_dims[-1] * self.size * self.size, self.cfg["model"]["latent_dim"]
        )

        # Build Decoder
        modules = []
        self.decoder_input = nn.Linear(
            self.cfg["model"]["latent_dim"], hidden_dims[-1] * self.size * self.size
        )
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(),
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims[-1],
                hidden_dims[-1],
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, self.final_dim, self.size, self.size)
        result = self.decoder(result)
        result = self.final_layer(result)
        result = self.celeb_transform1(result)
        result = torch.flatten(result, start_dim=1)
        result = torch.nan_to_num(result)
        return result

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var
