import os
import sys
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sbiu.unit import SBIU

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --- 1. Dataset Generation (Strictly 2Hz, 3Hz, 5Hz) ---
def generate_toy_data():
    """
    Time = 200 seconds. Fs = 10 Hz. Total 2000 points.
    Regimes: 2 Hz, 3 Hz, 5 Hz.
    """
    t = np.linspace(0, 200, 2000)
    y = np.zeros_like(t)
    labels = np.zeros(2000, dtype=int)
    
    # Generate stable transitions (mean hold time = 30 seconds = 300 points)
    switch_times = [0.0]
    while switch_times[-1] < 200.0:
        switch_times.append(switch_times[-1] + np.random.exponential(scale=35.0))
        
    current_regime = np.random.choice([0, 1, 2])
    regimes = [current_regime]
    for _ in range(1, len(switch_times)):
        choices = [r for r in [0, 1, 2] if r != current_regime]
        current_regime = np.random.choice(choices)
        regimes.append(current_regime)
        
    switch_idx = 0
    for i, ct in enumerate(t):
        if switch_idx + 1 < len(switch_times) and ct >= switch_times[switch_idx + 1]:
            switch_idx += 1
        regime = regimes[switch_idx]
        labels[i] = regime
        
        # Pure waves (No decay to ensure stationarity over 200s)
        if regime == 0:   y[i] = np.sin(4 * np.pi * ct)       # 2 Hz
        elif regime == 1: y[i] = np.cos(6 * np.pi * ct + 1.0) # 3 Hz
        elif regime == 2: y[i] = np.sin(10 * np.pi * ct)      # 5 Hz
            
    y += np.random.normal(0, 0.05, 2000)
    return t, y, labels

class PureSpectralWindowDataset(Dataset):
    def __init__(self, y, labels, window_size=10):
        self.window_size = window_size
        self.y = torch.tensor(y, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.y) - self.window_size
        
    def __getitem__(self, idx):
        window = self.y[idx : idx + self.window_size]
        fft_mag = torch.abs(torch.fft.rfft(window, dim=-1)) # Shape (6,)
        label = self.labels[idx + self.window_size // 2]
        return fft_mag, fft_mag, label

# --- 2. Pure Spectral Autoencoder Model (No Cheating Linear Encoder!) ---
class PureSpectralSBIU(nn.Module):
    def __init__(self, dim=6, init_scale=0.1): # Larger init std to allow free rotation
        super().__init__()
        self.sbiu = SBIU(dim=dim, shape=())
        # Apply standard initialization to start non-trivially
        nn.init.normal_(self.sbiu.basis.h_params, mean=0.0, std=init_scale)
        # Decoder is the only linear layer, used purely for reconstruction
        self.decoder = nn.Linear(dim, dim)
        
    def forward(self, x_fft, return_probs=False):
        # We normalize the raw FFT bins directly and pass them into the SBIU!
        state = nn.functional.normalize(x_fft, p=2, dim=-1)
        _, probs = self.sbiu(state, return_probs=True, pre_encoded=True)
        recon = self.decoder(probs)
        
        if return_probs: return recon, probs
        return recon

def main():
    print("=" * 60)
    print("RUNNING PURE SPECTRAL AUTOENCODER REGIME DISCOVERY")
    print("=" * 60)
    set_seed(42)
    os.makedirs("paper/figures", exist_ok=True)
    
    t, y, labels = generate_toy_data()
    dataset = PureSpectralWindowDataset(y, labels, window_size=10) # 10-step window
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = PureSpectralSBIU(dim=6, init_scale=0.15)
    # Give the basis more learning rate (0.002) so it is forced to actively rotate
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.MSELoss()
    
    print("Training Model (Unsupervised Spectral Reconstruction)...")
    model.train()
    for epoch in range(200):
        for X, Y, _ in dataloader:
            optimizer.zero_grad()
            loss = criterion(model(X), Y)
            loss.backward()
            optimizer.step()

    # --- 3. Evaluate and Generate Figure 1 ---
    model.eval()
    X_all = torch.stack([dataset[i][0] for i in range(len(dataset))])
    labels_all = np.array([dataset[i][2] for i in range(len(dataset))])
    
    with torch.no_grad():
        _, probs = model(X_all, return_probs=True)
        probs_np = probs.numpy().squeeze()
        
    preds = np.argmax(probs_np, axis=-1)
    print(f"Unsupervised ARI Score: {adjusted_rand_score(labels_all, preds):.4f}")
    
    # Plot Figure 1 (With fixed color indexes!)
    plt.figure(figsize=(10, 4.5))
    t_plot = t[5 : -5]
    # Fixed background color mapping: Index 0 (2Hz)=Green, Index 1 (3Hz)=Pink, Index 2 (5Hz)=Blue
    colors = ['#ccffcc', '#ffcccc', '#ccccff']
    
    start_idx = 0
    for i in range(1, len(labels_all)):
        if labels_all[i] != labels_all[i-1] or i == len(labels_all) - 1:
            plt.axvspan(t_plot[start_idx], t_plot[i], color=colors[labels_all[start_idx]], alpha=0.5, lw=0)
            start_idx = i
            
    world_colors = plt.cm.tab10(np.linspace(0, 1, 6))
    for m in range(6):
        if np.max(probs_np[:, m]) > 0.15: # Filter unused/zero-active states
            plt.plot(t_plot, probs_np[:, m], label=f'World {m+1}', color=world_colors[m], lw=2)
            
    plt.xlim(t_plot[0], t_plot[-1])
    plt.ylim(0, 1.05)
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Probability $p_m(t)$', fontsize=14)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=6)
    plt.tight_layout()
    plt.savefig("paper/figures/toy_world_probabilities.pdf")
    plt.close()
    print("Saved Figure 1 (Orthogonal State Probabilities).")
    
    # --- 4. Generate Figure 2 (Tuning Curves via True Activation Probability) ---
    test_freqs = np.linspace(0.5, 5.0, 300)
    tuning_activations = []
    
    with torch.no_grad():
        t_window = torch.linspace(0, 1.0, 10).unsqueeze(0)
        for freq in test_freqs:
            sine_window = torch.sin(2 * torch.pi * freq * t_window)
            fft_mag = torch.abs(torch.fft.rfft(sine_window, dim=-1))
            _, p = model(fft_mag, return_probs=True)
            tuning_activations.append(p.squeeze().numpy())
            
    tuning_activations = np.array(tuning_activations)
    
    plt.figure(figsize=(8, 4.5))
    for m in range(6):
        if np.max(tuning_activations[:, m]) > 0.15:
            plt.plot(test_freqs, tuning_activations[:, m], label=f'Pointer State {m+1}', color=world_colors[m], lw=2)
            
    plt.xlabel('Input Frequency (Hz)', fontsize=14)
    plt.ylabel('Activation Probability', fontsize=14)
    plt.xticks(np.arange(0, 6, 1))
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("paper/figures/pointer_tuning_curves.pdf")
    plt.close()
    print("Saved Figure 2 (Tuning Curves of Learned Pointer States).")

    # --- 5. Generate Figure 3: Heatmap of Learned Orthogonal Matrix V ---
    V = model.sbiu.basis.get_V().squeeze().cpu().detach().numpy()
    plt.figure(figsize=(7, 5))
    plt.imshow(np.abs(V), cmap='viridis', aspect='auto')
    plt.colorbar(label='Orthogonal Weight Magnitude')
    plt.xlabel('Pointer States (Orthogonal Modes)', fontsize=12)
    plt.ylabel('Base Fourier Features', fontsize=12)
    plt.xticks(np.arange(6), np.arange(1, 7))
    plt.yticks(np.arange(6), ['0 Hz DC', '1 Hz Cos/Sin', '2 Hz Cos/Sin', '3 Hz Cos/Sin', '4 Hz Cos/Sin', '5 Hz Cos/Sin'])
    plt.tight_layout()
    plt.savefig("paper/figures/pointer_eigenbasis_heatmap.pdf")
    plt.close()
    print("Saved Figure 3 (Non-trivial, rotated orthogonal Heatmap).")
    print("=" * 60)

if __name__ == "__main__":
    main()