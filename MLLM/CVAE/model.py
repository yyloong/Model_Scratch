import torch.nn as nn
import torch

class CVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, num_classes):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1 + num_classes, 32, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Flatten(),
            nn.Linear(32 * input_dim * input_dim, latent_dim * 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 32 * input_dim * input_dim),
            nn.ReLU(),
            nn.Unflatten(1, (32, input_dim, input_dim)),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def add_condition(self, x, c):
        c = c.view(c.size(0), -1, 1, 1)
        c = c.expand(-1, -1, x.size(2), x.size(3))
        return torch.cat([x, c], dim=1)
    
    def forward(self, x, c):
        x = self.add_condition(x, c)
        enc_out = self.encoder(x)
        mu, logvar = torch.chunk(enc_out, 2, dim=1)
        z = self.reparameterize(mu, logvar)
        z = torch.cat([z, c], dim=1)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar
