import torch
from torch.utils.data import DataLoader, random_split, TensorDataset
from config import *

import torch
from torch.utils.data import DataLoader, random_split, TensorDataset
from config import *


import numpy as np
import torch
import geomstats.backend as gs
from geomstats.geometry.spd_matrices import SPDMatrices
from config import *

def generate_spd_matrix(n_samples=1, dim=3):
    """Generate SPD matrices using geomstats"""
    manifold = SPDMatrices(n=dim)
    # Generate random SPD matrices using geomstats
    spd_matrices = manifold.random_point(n_samples=n_samples)
    # Convert to torch tensor
    return torch.tensor(spd_matrices, dtype=torch.float32)

def generate_spd_class(mean_matrix, n_samples=1000):
    """Generate a class of SPD matrices with specified mean"""
    manifold = SPDMatrices(n=mean_matrix.shape[0])
    # Generate perturbations around mean_matrix
    perturbations = manifold.random_tangent_vec(base_point=mean_matrix.numpy(), n_samples=n_samples)
    # Exponential map to get SPD matrices
    spd_matrices = manifold.exp(tangent_vec=perturbations, base_point=mean_matrix.numpy())
    return torch.tensor(spd_matrices, dtype=torch.float32)

def generate_mixed_spd_data(class1_samples=1000, class2_samples=1000, mix_ratio=0.0):
    """Generate two classes of SPD matrices with mixing"""
    manifold = SPDMatrices(n=3)
    
    # Generate two distinct mean matrices
    mean1 = torch.eye(3)
    mean2 = torch.diag(torch.tensor([2.0, 2.0, 2.0]))
    
    # Generate base classes
    class1 = generate_spd_class(mean1, class1_samples)
    class2 = generate_spd_class(mean2, class2_samples)
    
    if mix_ratio > 0:
        # Use geomstats interpolation for mixing
        mix_samples = int(mix_ratio * min(class1_samples, class2_samples))
        interpolated = manifold.geodesic(initial_point=mean1.numpy(), 
                                       end_point=mean2.numpy())(gs.random.rand(mix_samples))
        interpolated = torch.tensor(interpolated, dtype=torch.float32)
        
        # Replace some samples with interpolated ones
        class1[:mix_samples] = interpolated
        class2[:mix_samples] = interpolated
    
    # Combine data and create labels
    data = torch.cat([class1, class2])
    labels = torch.cat([torch.zeros(class1_samples), torch.ones(class2_samples)])
    
    return data, labels

def generate_spd_data(num_samples=1000):
    """Generate random SPD matrices for training"""
    return generate_spd_matrix(n_samples=num_samples)

def get_dataloaders(data):
    """Process data and create dataloaders"""
    if len(data.shape) == 3:  # If data is (N, 3, 3)
        N = data.shape[0]
        data = data.reshape(N, -1)  # Flatten to (N, 9)
    
    dataset = TensorDataset(data)
    # Split sizes
    train_size = int(TRAIN_RATIO * len(dataset))
    val_size = int(VAL_RATIO * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_data, val_data, test_data = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    return (
        DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(val_data, batch_size=BATCH_SIZE),
        DataLoader(test_data, batch_size=BATCH_SIZE)
    )