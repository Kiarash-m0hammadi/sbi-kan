import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

# Ensure root directory is in path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from eeg_discovery import EEGDataset, EEGSpectralAutoencoder, set_seed

def analyze_seizure_split(model, dataset, device):
    """
    Analyzes the physical/frequency difference between the split seizure worlds.
    """
    model.eval()
    X_all = dataset.X.to(device)
    labels_all = dataset.y.numpy()
    
    with torch.no_grad():
        _, probs, x_fft = model(X_all, return_probs=True)
        probs = probs.cpu().numpy()
        x_fft = x_fft.cpu().numpy()
        
    predicted_clusters = np.argmax(probs, axis=-1)
    
    # Extract only the actual seizure rows (True Label 0)
    seizure_indices = np.where(labels_all == 0)[0]
    seizure_preds = predicted_clusters[seizure_indices]
    seizure_ffts = x_fft[seizure_indices]
    
    # Find the top 2 worlds where seizures were routed
    unique_worlds, counts = np.unique(seizure_preds, return_counts=True)
    sorted_idx = np.argsort(counts)[::-1]
    
    print("\n" + "="*60)
    print("PHYSICAL SPECTRAL ANALYSIS OF THE SEIZURE SPLIT")
    print("="*60)
    
    if len(sorted_idx) < 2:
        print("Seizures did not split significantly. Skipping spectral analysis.")
        return
        
    for i in range(2):
        world_num = unique_worlds[sorted_idx[i]]
        world_count = counts[sorted_idx[i]]
        
        # Get FFTs of seizures assigned to this world
        world_ffts = seizure_ffts[seizure_preds == world_num]
        mean_fft = np.mean(world_ffts, axis=0)
        
        # In our dataset (178 Hz, 178 samples), FFT bin k corresponds EXACTLY to k Hz.
        # Find the peak frequencies (excluding DC offset at index 0)
        peak_bins = np.argsort(mean_fft[1:])[::-1][:3] + 1 # Top 3 peaks
        peak_freqs = peak_bins # Since 1 bin = 1 Hz
        
        # Calculate total energy (mean spectral magnitude)
        total_energy = np.mean(mean_fft)
        
        print(f"Seizure Sub-Regime in World {world_num} ({world_count} seizures):")
        print(f"  - Peak Frequency Channels: {peak_freqs[0]} Hz, {peak_freqs[1]} Hz, {peak_freqs[2]} Hz")
        print(f"  - Mean Spectral Energy: {total_energy:.4f}")
        print("-" * 40)
        
    print("Interpretation:")
    print("If the peak frequencies or energies differ, the SBIU successfully")
    print("sub-classified the seizures into different physiological phases")
    print("(e.g., slow-wave spike activity vs. rapid tonic firing) completely unsupervised!")

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Try dim=12 (12 Worlds)
    dim_to_test = 12
    print(f"\nLaunching Investigation with d={dim_to_test} Worlds...")
    
    csv_path = 'src/data/eeg/Epileptic Seizure Recognition.csv'
    dataset = EEGDataset(csv_path)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    model = EEGSpectralAutoencoder(in_features=178, dim=dim_to_test, init_scale=0.01).to(device)
    
    basis_params = [p for n, p in model.named_parameters() if 'basis' in n]
    other_params = [p for n, p in model.named_parameters() if 'basis' not in n]
    optimizer = torch.optim.Adam([
        {'params': other_params, 'lr': 0.005, 'weight_decay': 1e-4},
        {'params': basis_params, 'lr': 0.0005, 'weight_decay': 0.0} 
    ])
    
    criterion = nn.MSELoss()
    
    # Train
    model.train()
    for epoch in range(150):
        for X, _ in dataloader:
            X = X.to(device)
            optimizer.zero_grad()
            recon, target_fft = model(X)
            loss = criterion(recon, target_fft)
            loss.backward()
            optimizer.step()
            
    # Audit
    model.eval()
    X_all = dataset.X.to(device)
    labels_all = dataset.y.numpy()
    
    with torch.no_grad():
        _, probs, _ = model(X_all, return_probs=True)
        probs = probs.cpu().numpy()
        
    predicted_clusters = np.argmax(probs, axis=-1)
    
    print("\n" + "="*60)
    print(f"AUDITING THE {dim_to_test}-WORLD SBIU MODEL")
    print("="*60)
    df_eval = pd.DataFrame({'True_Class': labels_all, 'SBIU_World': predicted_clusters})
    class_names = {0: "1 - Seizure", 1: "2 - Tumor Area", 2: "3 - Healthy Area", 3: "4 - Eyes Closed", 4: "5 - Eyes Open"}
    df_eval['True_Class'] = df_eval['True_Class'].map(class_names)
    crosstab = pd.crosstab(df_eval['SBIU_World'], df_eval['True_Class'])
    print(crosstab)
    
    # Run the physical spectral analysis
    analyze_seizure_split(model, dataset, device)

if __name__ == "__main__":
    main()