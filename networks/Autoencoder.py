import torch
from torch import nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim=80, latent_dim=10):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Dropout(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, latent_dim)

            # alternative encoder
            # nn.Linear(input_dim, 32),
            # nn.BatchNorm1d(32),
            # nn.ReLU(),
            # nn.Linear(32, 16),
            # nn.BatchNorm1d(16),
            # nn.ReLU(),
            # nn.Linear(16, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, input_dim)

            # alternative decoder
            # nn.Linear(latent_dim, 16),
            # nn.BatchNorm1d(16),
            # nn.ReLU(),
            # nn.Linear(16, 32),
            # nn.BatchNorm1d(32),
            # nn.ReLU(),
            # nn.Linear(32, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, z
