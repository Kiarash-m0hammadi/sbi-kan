import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_self_supervised(model: nn.Module, 
                          dataloader: DataLoader, 
                          optimizer: torch.optim.Optimizer, 
                          epochs: int, 
                          device: torch.device,
                          verbose: bool = True) -> list:
    """
    Trains the SBI-KAN (or baseline) using the self-supervised next-step prediction loss.
    
    Paper Reference: Section 7.3 "Training Objective" (Mean Squared Error).
    """
    model.to(device)
    criterion = nn.MSELoss()
    epoch_losses = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        # Determine if we should show a progress bar
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}") if verbose else dataloader
        
        for batch in pbar:
            # Our datasets yield (X, Y, label). We ignore label during training.
            X, Y, _ = batch
            X, Y = X.to(device), Y.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            # We don't need return_probs=True during training, saving memory
            preds = model(X)
            
            # Loss: MSE between prediction and next-step
            loss = criterion(preds, Y)
            loss.backward()
            
            optimizer.step()
            
            total_loss += loss.item() * X.size(0)
            
            if verbose:
                pbar.set_postfix({"MSE": loss.item()})
                
        avg_loss = total_loss / len(dataloader.dataset)
        epoch_losses.append(avg_loss)
        
        if not verbose:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")
            
    return epoch_losses