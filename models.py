import torch.nn as nn
from config import *

class SPDEncoder(nn.Module):
    """Maps 9D SPD vectors to latent space with dimension reduction"""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(INPUT_DIM, 8),
            nn.ReLU(),
            nn.Linear(8, 6),
            nn.ReLU(),
            nn.Linear(6, 4),
            nn.ReLU(),
            nn.Linear(4, LATENT_DIM)
        )
        
    def forward(self, x):
        return self.layers(x)

class SPDDecoder(nn.Module):
    """Maps latent space to SPD matrices with symmetry + positive definiteness"""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(LATENT_DIM, 4),
            nn.ReLU(),
            nn.Linear(4, 6),
            nn.ReLU(),
            nn.Linear(6, 8),
            nn.ReLU(),
            nn.Linear(8, 9)
        )
        
    def forward(self, z):
        x = self.layers(z)
        x = x.view(-1, 3, 3)
        # Enforce symmetry and positive definiteness
        x = 0.5 * (x + x.transpose(1, 2))  # Symmetric
        x += torch.eye(3, device=x.device) * SPD_EPS  # PD guarantee
        return x.flatten(1)

class SPDAutoencoder(nn.Module):
    """Full autoencoder with SPD constraints"""
    def __init__(self):
        super().__init__()
        self.encoder = SPDEncoder()
        self.decoder = SPDDecoder()
        
    def forward(self, x):
        return self.decoder(self.encoder(x))