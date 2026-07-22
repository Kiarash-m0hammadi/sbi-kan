import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from sbiu.unit import SBIU
from src.experiments.eeg_discovery import set_seed

# --- 1. Dataset Loader for MIT-BIH ---
class ECGDataset(Dataset):
    def __init__(self, csv_path):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Could not find {csv_path}. Please place mitbih_train.csv in src/data/")
            
        print(f"Reading {csv_path}...")
        df = pd.read_csv(csv_path, header=None)
        
        # 187 signal points
        X_raw = df.iloc[:, :187].values
        # The 188th column is the label
        y_raw = df.iloc[:, 187].values.astype(int)
        
        # Instance Normalization (Standardize each heartbeat individually)
        X_mean = X_raw.mean(axis=1, keepdims=True)
        X_std = X_raw.std(axis=1, keepdims=True) + 1e-8
        X_raw = (X_raw - X_mean) / X_std
        
        self.X = torch.tensor(X_raw, dtype=torch.float32)
        self.y = torch.tensor(y_raw, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- 2. ECG Spectral Autoencoder ---
class ECGSpectralAutoencoder(nn.Module):
    def __init__(self, in_features=187, dim=8, init_scale=0.01):
        super().__init__()
        # Real FFT of 187 points yields (187 // 2) + 1 = 94 frequency bins
        self.fft_dim = in_features // 2 + 1 
        
        self.projection = nn.Linear(self.fft_dim, dim)
        self.sbiu = SBIU(dim=dim, shape=())
        nn.init.normal_(self.sbiu.basis.h_params, mean=0.0, std=init_scale)
        self.decoder = nn.Linear(dim, self.fft_dim)
        
    def forward(self, x, return_probs=False):
        x_fft = torch.abs(torch.fft.rfft(x, dim=-1))
        x_fft = torch.log1p(x_fft)  # Log spectral transform
        
        state = self.projection(x_fft)
        state = nn.functional.normalize(state, p=2, dim=-1)
        
        _, probs = self.sbiu(state, return_probs=True, pre_encoded=True)
        recon = self.decoder(probs)
        
        if return_probs:
            return recon, probs, x_fft
        return recon, x_fft

# --- 3. Training Loop ---
def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    csv_path = 'src/data/ecg/mitbih_train.csv' 
    print("\n--- Loading ECG Dataset ---")
    dataset = ECGDataset(csv_path)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    # Bottleneck dimension of 8 (giving coordinate headroom for 5 clinical classes)
    model = ECGSpectralAutoencoder(in_features=187, dim=8, init_scale=0.01).to(device)
    
    # Discriminative Learning Rates
    basis_params = [p for n, p in model.named_parameters() if 'basis' in n]
    other_params = [p for n, p in model.named_parameters() if 'basis' not in n]
    optimizer = torch.optim.Adam([
        {'params': other_params, 'lr': 0.005, 'weight_decay': 1e-4},
        {'params': basis_params, 'lr': 0.0005, 'weight_decay': 0.0} 
    ])
    
    criterion = nn.MSELoss()
    
    print("\n--- Training Unsupervised ECG Bottleneck ---")
    model.train()
    epochs = 80 
    pbar = tqdm(range(epochs), desc="Training Autoencoder")
    for epoch in pbar:
        total_loss = 0
        for X, _ in dataloader:
            X = X.to(device)
            optimizer.zero_grad()
            recon, target_fft = model(X)
            loss = criterion(recon, target_fft)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({"MSE Loss": f"{total_loss/len(dataloader):.4f}"})

    # Save the trained model weights
    os.makedirs('src/models', exist_ok=True)
    torch.save(model.state_dict(), 'src/models/ecg_sbiu.pt')
    print("\n[INFO] Model successfully saved to src/models/ecg_sbiu.pt")

    # --- 4. Unsupervised Evaluation & Audit ---
    print("\n--- Evaluating ECG Regimes ---")
    model.eval()
    X_all = dataset.X.to(device)
    labels_all = dataset.y.numpy()
    
    with torch.no_grad():
        _, probs, _ = model(X_all, return_probs=True)
        probs = probs.cpu().numpy()
        
    predicted_clusters = np.argmax(probs, axis=-1)
    
    print("\n" + "="*70)
    print("AUDITING THE ECG SBIU WORLDS")
    print("="*70)
    df_eval = pd.DataFrame({'True_Class': labels_all, 'SBIU_World': predicted_clusters})
    class_names = {
        0: "0 - Normal (N)", 
        1: "1 - Supraventricular (S)", 
        2: "2 - Ventricular (V)", 
        3: "3 - Fusion (F)", 
        4: "4 - Unclassified (Q)"
    }
    df_eval['True_Class'] = df_eval['True_Class'].map(class_names)
    crosstab = pd.crosstab(df_eval['SBIU_World'], df_eval['True_Class'])
    print(crosstab)
    print("="*70)

if __name__ == "__main__":
    main()