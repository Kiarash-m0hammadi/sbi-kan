import numpy as np
import torch
from torch.utils.data import Dataset

def generate_transient_data(n_points=2000, mean_hold_time=0.15, noise_std=0.05, seed=42):
    """
    Generates a signal with two regimes:
    Regime 0: Smooth 30Hz oscillation.
    Regime 1: 30Hz oscillation + sharp transient spikes (every 0.05s).
    """
    if seed is not None:
        np.random.seed(seed)
        
    t = np.linspace(0, 1, n_points)
    y = np.zeros_like(t)
    labels = np.zeros(n_points, dtype=int)
    
    switch_times = [0.0]
    while switch_times[-1] < 1.0:
        hold_time = np.random.exponential(scale=mean_hold_time)
        switch_times.append(switch_times[-1] + hold_time)
        
    current_regime = np.random.choice([0, 1])
    regimes_at_switches = [current_regime]
    for _ in range(1, len(switch_times)):
        current_regime = 1 - current_regime
        regimes_at_switches.append(current_regime)
        
    switch_idx = 0
    for i, current_t in enumerate(t):
        if switch_idx + 1 < len(switch_times) and current_t >= switch_times[switch_idx + 1]:
            switch_idx += 1
        regime = regimes_at_switches[switch_idx]
        labels[i] = regime
        
        # Base smooth oscillation (30 Hz)
        base_wave = np.sin(60 * np.pi * current_t)
        
        if regime == 0:
            y[i] = base_wave
        elif regime == 1:
            # Add a sharp transient spike (duration of ~2 points) every 0.05s
            spike = 0
            if (current_t % 0.05) < 0.0015:
                spike = 3.0
            y[i] = base_wave + spike
            
    y += np.random.normal(loc=0.0, scale=noise_std, size=n_points)
    return t, y, labels

class TransientWindowDataset(Dataset):
    def __init__(self, y: np.ndarray, labels: np.ndarray, window_size: int = 50):
        self.window_size = window_size
        self.y = torch.tensor(y, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.y) - self.window_size
        
    def __getitem__(self, idx):
        X = self.y[idx : idx + self.window_size]
        Y = self.y[idx + self.window_size].unsqueeze(-1)
        label = self.labels[idx + self.window_size]
        return X, Y, label