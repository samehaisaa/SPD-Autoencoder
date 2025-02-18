import torch.nn as nn
from config import INPUT_DIM, LATENT_DIM

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