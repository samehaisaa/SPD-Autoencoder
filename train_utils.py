import torch
from torch import optim
from config import *
from models import SPDAutoencoder
import torch.nn as nn
def initialize_model(lr=LEARNING_RATE):
    """Model initialization with proper weight seeding"""
    torch.manual_seed(SEED)
    model = SPDAutoencoder().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    return model, optimizer, criterion

def train_epoch(model, loader, optimizer, criterion):
    """Single training epoch"""
    model.train()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        inputs = batch[0].to(DEVICE)
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
    return total_loss / len(loader.dataset)

def evaluate_model(model, test_loader, criterion):
    """
    Comprehensive model evaluation returning test loss and other metrics
    """
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch[0].to(DEVICE)
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, inputs)
            total_loss += loss.item() * inputs.size(0)
            
            # Store predictions and targets
            all_predictions.append(outputs.cpu())
            all_targets.append(inputs.cpu())
    
    # Calculate average loss
    avg_loss = total_loss / len(test_loader.dataset)
    
    # Combine all batches
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Calculate reconstruction error (MSE)
    reconstruction_error = torch.mean((all_predictions - all_targets) ** 2).item()
    
    return {
        'test_loss': avg_loss,
        'reconstruction_error': reconstruction_error,
        'predictions': all_predictions,
        'targets': all_targets
    }
    
    
def full_train(model, train_loader, val_loader, optimizer, criterion, epochs=EPOCHS):
    """Complete training loop with early stopping"""
    best_loss = float('inf')
    best_weights = None
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss = evaluate(model, val_loader, criterion)
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4e} | Val Loss: {val_loss:.4e}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = model.state_dict()
    
    model.load_state_dict(best_weights)
    return model


def load_or_train_model(train_loader, val_loader, model_path='model.pt'):
    """Load a trained model or train a new one if no saved model exists"""
    try:
        model = SPDAutoencoder().to(DEVICE)
        model.load_state_dict(torch.load(model_path,weights_only=False))
        print("Loaded trained model")
        return model
    except FileNotFoundError:
        print("No saved model found. Training new model...")
        model, optimizer, criterion = initialize_model()
        model = full_train(model, train_loader, val_loader, optimizer, criterion)
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        return model