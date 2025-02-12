import matplotlib.pyplot as plt
import numpy as np
import os
from config import *

# Create a directory for saving plots
os.makedirs("plots", exist_ok=True)

def get_versioned_filename(base_name, extension="png"):
    """
    Generates a versioned filename (e.g., base_name_1.png, base_name_2.png, etc.).
    """
    version = 1
    while True:
        filename = f"plots/{base_name}_{version}.{extension}"
        if not os.path.exists(filename):
            return filename
        version += 1

def plot_latent_with_classes(z, labels, mix_ratio):
    """2D latent space visualization with class coloring"""
    plt.figure(figsize=(10, 8))
    z_np = z.cpu().numpy()
    
    # Plot with class colors
    scatter = plt.scatter(z_np[:,0], z_np[:,1], c=labels.cpu().numpy(),
                         cmap='viridis', alpha=0.6, edgecolors='w', s=50)
    
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.title(f"Latent Space (Mix Ratio: {mix_ratio})")
    plt.colorbar(scatter, label='Class')
    plt.grid(True)
    
    # Save with versioning
    filename = f"latent_mix_{int(mix_ratio*100):03d}.png"
    plt.savefig(os.path.join("plots", filename), bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved {filename}")

def error_distribution(model, loader):
    """Reconstruction error analysis with versioned filenames"""
    filename = get_versioned_filename("error_distribution")
    errors = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            inputs = batch[0].to(DEVICE)
            outputs = model(inputs)
            err = torch.mean((outputs - inputs)**2, dim=1)
            errors.extend(err.cpu().numpy())
    
    plt.figure(figsize=(8,4))
    plt.hist(errors, bins=25, density=True, alpha=0.7)
    plt.xlabel("Reconstruction MSE")
    plt.ylabel("Probability Density")
    plt.title("Error Distribution")
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved error distribution plot to {filename}")

def matrix_analysis(original, reconstructed):
    """Comparative SPD matrix analysis with versioned filenames"""
    filename = get_versioned_filename("matrix_analysis")
    fig, ax = plt.subplots(1, 3, figsize=(15,5))
    
    for i, (mat, title) in enumerate(zip([original, reconstructed, original-reconstructed], 
                                       ["Original", "Reconstructed", "Difference"])):
        im = ax[i].imshow(mat, cmap='viridis')
        ax[i].set_title(title)
        fig.colorbar(im, ax=ax[i])
    
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved matrix analysis plot to {filename}")
    
    # Eigenvalue analysis
    eig_orig = np.linalg.eigvalsh(original)
    eig_recon = np.linalg.eigvalsh(reconstructed)
    
    print(f"Original Eigenvalues: {eig_orig}")
    print(f"Reconstructed Eigenvalues: {eig_recon}")
    print(f"Symmetric Check (Original): {np.allclose(original, original.T)}")
    print(f"Symmetric Check (Reconstructed): {np.allclose(reconstructed, reconstructed.T)}")