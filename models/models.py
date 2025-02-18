import torch.nn as nn
from .encoder import SPDEncoder
from .decoder import SPDDecoder

class SPDAutoencoder(nn.Module):
    """Full autoencoder with SPD constraints"""
    def __init__(self):
        super().__init__()
        self.encoder = SPDEncoder()
        self.decoder = SPDDecoder()
        
    def forward(self, x):
        return self.decoder(self.encoder(x))