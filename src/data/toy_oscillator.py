import numpy as np
import torch
from torch.utils.data import Dataset

def generate_toy_data(n_points: int = 2000, 
                      mean_hold_time: float = 0.15, 
                      noise_std: float = 0.05, 
                      seed: int = None,
                      freq_multiplier: float = 1.0):
    """
    Generates the synthetic hidden-regime oscillator data as described 
    in Section 8.1.1 of the paper.
    
    Args:
        n_points (int): Number of points in the signal.
        mean_hold_time (float): Mean regime duration in seconds.
        noise_std (float): Standard deviation of Gaussian noise.
        seed (int, optional): Random seed.
        freq_multiplier (float): Multiplier to scale the frequency of the oscillators.
                                 Defaults to 1.0. Set to 15.0 to break local next-step 
                                 shortcuts and force global spectral learning.
    
    Regimes (unscaled):
        A (0): y = sin(4 * pi * t)
        B (1): y = cos(6 * pi * t + pi / 3)
        C (2): y = sin(10 * pi * t) * exp(-2t)
    
    Returns:
        t (np.ndarray): Time array
        y (np.ndarray): Noisy signal array
        labels (np.ndarray): Ground truth regime labels (0, 1, or 2)
    """
    if seed is not None:
        np.random.seed(seed)

    t = np.linspace(0, 1, n_points)
    y = np.zeros_like(t)
    labels = np.zeros(n_points, dtype=int)

    # 1. Generate transition times
    switch_times = [0.0]
    while switch_times[-1] < 1.0:
        hold_time = np.random.exponential(scale=mean_hold_time)
        switch_times.append(switch_times[-1] + hold_time)

    # 2. Generate regime sequence ensuring no consecutive repeats
    current_regime = np.random.choice([0, 1, 2])
    regimes_at_switches = [current_regime]

    for _ in range(1, len(switch_times)):
        choices = [r for r in [0, 1, 2] if r != current_regime]
        current_regime = np.random.choice(choices)
        regimes_at_switches.append(current_regime)

    # 3. Assign regimes and compute raw signal
    switch_idx = 0
    for i, current_t in enumerate(t):
        if switch_idx + 1 < len(switch_times) and current_t >= switch_times[switch_idx + 1]:
            switch_idx += 1
            
        regime = regimes_at_switches[switch_idx]
        labels[i] = regime
        
        # Apply the corresponding regime's function scaled by the multiplier
        if regime == 0:    # Regime A
            y[i] = np.sin(4 * np.pi * freq_multiplier * current_t)
        elif regime == 1:  # Regime B
            y[i] = np.cos(6 * np.pi * freq_multiplier * current_t + np.pi / 3)
        elif regime == 2:  # Regime C
            y[i] = np.sin(10 * np.pi * freq_multiplier * current_t) * np.exp(-2 * current_t)

    # 4. Add Gaussian white noise
    y += np.random.normal(loc=0.0, scale=noise_std, size=n_points)

    return t, y, labels


class AbsoluteTimeDataset(Dataset):
    """
    Variant 1: Absolute time input (Section 8.1.1)
    Receives scalar t as input, predicts y(t).
    """
    def __init__(self, t: np.ndarray, y: np.ndarray, labels: np.ndarray):
        self.t = torch.tensor(t, dtype=torch.float32).unsqueeze(-1)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.t)
        
    def __getitem__(self, idx):
        return self.t[idx], self.y[idx], self.labels[idx]


class SlidingWindowDataset(Dataset):
    """
    Variant 2: Sliding window input (Section 8.1.1)
    Receives window (y_{i-L+1}, ..., y_i), predicts y_{i+1+gap}.
    By setting gap > 0, we force the model to learn the global frequency
    to project the wave into the future, breaking local shortcuts.
    """
    def __init__(self, y: np.ndarray, labels: np.ndarray, window_size: int = 50, gap: int = 25):
        self.window_size = window_size
        self.gap = gap
        self.y = torch.tensor(y, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.y) - self.window_size - self.gap
        
    def __getitem__(self, idx):
        # Input features: window of length L
        X = self.y[idx : idx + self.window_size]
        # Target: the time step after the gap
        Y = self.y[idx + self.window_size + self.gap].unsqueeze(-1)
        # Ground truth regime of the target
        label = self.labels[idx + self.window_size + self.gap]
        
        return X, Y, label