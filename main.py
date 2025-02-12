import torch
import torch.nn as nn
from data_utils import generate_spd_class,generate_spd_data, get_dataloaders,generate_mixed_spd_data
from models import SPDAutoencoder
from train_utils import initialize_model, full_train, evaluate_model,load_or_train_model
from visualization import  error_distribution, matrix_analysis,plot_latent_with_classes
from config import *
from torch.utils.data import TensorDataset, DataLoader






def run_experiment(model, mix_ratio, num_samples=1001):
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
            print(z)
            latent.append(z.cpu())
    
    latent = torch.cat(latent)
    plot_latent_with_classes(latent, labels, mix_ratio)









def main():
    train_loader, val_loader, test_loader = get_dataloaders(generate_spd_data())
    model = load_or_train_model(train_loader, val_loader, model_path='model.pt')
    for mix_ratio in [0.0, 0.3, 0.6, 0.9]:
        print(f"\nRunning experiment with mix ratio: {mix_ratio}")
        run_experiment(model, mix_ratio)
        
    print("All experiments completed!")

    

if __name__ == "__main__":
    torch.manual_seed(SEED)
    main()