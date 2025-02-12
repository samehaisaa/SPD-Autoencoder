import torch.nn as nn
from config import *

import torch
import torch.nn as nn

class SPDEncoder(nn.Module):
    def __init__(self, input_dim=9):  # 3x3 matrix flattened = 9
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 6),
            nn.ReLU(),
            nn.Linear(6, 2)  # 2D latent space
        )
    
    def forward(self, x):
        # Reshape input if necessary
        if len(x.shape) == 3:  # If input is batch of matrices
            batch_size = x.shape[0]
            x = x.view(batch_size, -1)  # Flatten each matrix
        return self.layers(x)

class SPDDecoder(nn.Module):
    def __init__(self, output_dim=9):  # 3x3 matrix flattened = 9
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 6),
            nn.ReLU(),
            nn.Linear(6, 8),
            nn.ReLU(),
            nn.Linear(8, output_dim)
        )
    
    def forward(self, x):
        x = self.layers(x)
        # Reshape output to matrix form
        batch_size = x.shape[0]
        return x.view(batch_size, 3, 3)

class SPDAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SPDEncoder()
        self.decoder = SPDDecoder()
    
    def forward(self, x):
        return self.decoder(self.encoder(x))