import torch
import torch.nn as nn
from config import LATENT_DIM, SPD_EPS

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
        # Enforce symmetry and PD
        x = 0.5 * (x + x.transpose(1, 2))  # Symmetric
        x += torch.eye(3, device=x.device) * SPD_EPS  # PD guarantee
        return x.flatten(1)