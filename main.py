import torch
import torch.nn as nn
from data_utils import generate_spd_data, get_dataloaders,generate_mixed_spd_data
from models import SPDAutoencoder
from train_utils import initialize_model, full_train, evaluate_model,load_or_train_model
from visualization import plot_latent, error_distribution, matrix_analysis,plot_latent_with_classes
from config import *
from tensorflow import TensorDataset, DataLoader







def run_experiment(model, mix_ratio, num_samples=500):
    # Generate data with specified mixing
    data, labels = generate_mixed_spd_data(
        class1_samples=num_samples,
        class2_samples=num_samples,
        mix_ratio=mix_ratio
    )
    
    # Get latent representations
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    latent = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            z = model.encoder(batch[0].to(DEVICE))
            latent.append(z.cpu())
    
    latent = torch.cat(latent)
    plot_latent_with_classes(latent, labels, mix_ratio)









def main():
    # Data pipeline
    data = generate_spd_data()
    train_loader, val_loader, test_loader = get_dataloaders(data)
    print(f"Train: {len(train_loader)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")
    # Model initialization
    model = load_or_train_model(train_loader, val_loader)
    
    # Evaluate model
    criterion = nn.MSELoss()
    eval_results = evaluate_model(model, test_loader, criterion)
    
    print("\nModel Evaluation Results:")
    print(f"Test Loss: {eval_results['test_loss']:.4e}")
    print(f"Reconstruction Error: {eval_results['reconstruction_error']:.4e}")
    
    # Latent space analysis
    with torch.no_grad():
        latent = [model.encoder(batch[0].to(DEVICE)) for batch in test_loader]
    latent = torch.cat(latent)
    plot_latent(latent)
    
    # Error analysis
    error_distribution(model, test_loader)
    
    # Sample reconstructions
    sample = next(iter(test_loader))[0][:5].to(DEVICE)
    with torch.no_grad():
        recon = model(sample)
        
    
    
    for mix_ratio in [0.0, 0.3, 0.6, 0.9]:
        print(f"\nRunning experiment with mix ratio: {mix_ratio}")
        run_experiment(model, mix_ratio)
        
    print("All experiments completed!")

    

if __name__ == "__main__":
    torch.manual_seed(SEED)
    main()