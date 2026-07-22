import math
import torch
import torch.nn as nn
import numpy as np

# Import your own encoding to ensure MLP-Fourier gets the EXACT same features
from sbiu.encoding import FourierEncoding 

# ==========================================
# 1. MLP-raw Baseline
# ==========================================
class MLPRaw(nn.Module):
    """
    Standard MLP that receives the raw inputs.
    Paper: "two hidden layers (64 ReLU units each)"
    """
    def __init__(self, in_features: int, out_features: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_features)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (..., in_features)
        return self.net(x)


# ==========================================
# 2. MLP-Fourier Baseline
# ==========================================
class MLPFourier(nn.Module):
    """
    MLP that receives the EXACT same Fourier features as the SBIU.
    Crucial for proving that the orthogonal constraints (and not just 
    the Fourier features) are responsible for interpretability.
    """
    def __init__(self, in_features: int, out_features: int, dim: int = 8, base_freq: float = 1.0, hidden_dim: int = 64):
        super().__init__()
        # Use the exact same encoding as SBIU
        self.encoding = FourierEncoding(dim=dim, base_freq=base_freq)
        
        # The encoding outputs (in_features, dim). We flatten this.
        encoded_dim = in_features * dim
        
        self.net = nn.Sequential(
            nn.Linear(encoded_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_features)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (..., in_features)
        encoded = self.encoding(x) # Shape: (..., in_features, dim)
        
        # Flatten the features and the dim into a single vector per batch item
        flattened = encoded.flatten(start_dim=-2) # Shape: (..., in_features * dim)
        
        return self.net(flattened)


# ==========================================
# 3. Fourier-KAN (FKAN) Baseline
# ==========================================
class FourierKANLinear(nn.Module):
    """
    A single 1D Fourier-KAN edge layer.
    Replaces B-splines with Fourier series: a_k cos(kx) + b_k sin(kx)
    """
    def __init__(self, in_features: int, out_features: int, grid_size: int = 8, add_bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.add_bias = add_bias
        
        # Trainable Fourier coefficients
        # Shape: (2, out_features, in_features, grid_size)
        # index 0 -> Cosine coefficients, index 1 -> Sine coefficients
        self.fourier_coeffs = nn.Parameter(
            torch.randn(2, out_features, in_features, grid_size) / 
            (math.sqrt(in_features) * grid_size)
        )
        
        if self.add_bias:
            self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (..., in_features)
        k = torch.arange(1, self.grid_size + 1, device=x.device, dtype=x.dtype)
        
        # Expand x against the frequencies: (..., in_features, grid_size)
        x_expanded = x.unsqueeze(-1) * k
        
        cos_feat = torch.cos(x_expanded)
        sin_feat = torch.sin(x_expanded)
        
        # Efficient sum over input features (i) and grid frequencies (g) -> out_features (o)
        y = torch.einsum('...ig,oig->...o', cos_feat, self.fourier_coeffs[0]) + \
            torch.einsum('...ig,oig->...o', sin_feat, self.fourier_coeffs[1])
            
        if self.add_bias:
            y += self.bias
            
        return y

class FourierKAN(nn.Module):
    """
    Full Fourier-KAN model matching the Kolmogorov-Arnold topology.
    """
    def __init__(self, layer_sizes: list, grid_size: int = 8):
        """
        layer_sizes: list of integers, e.g., [in_features, hidden_features, out_features]
        """
        super().__init__()
        self.layers = nn.ModuleList([
            FourierKANLinear(in_dim, out_dim, grid_size=grid_size)
            for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:])
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# ==========================================
# 4. Standard Spline-KAN (efficient-kan)
# ==========================================
class SplineKAN(nn.Module):
    """
    Wrapper for the standard B-spline KAN.
    Requires `efficient-kan` installed via pip.
    """
    def __init__(self, layer_sizes: list, grid_size: int = 5, spline_order: int = 3):
        super().__init__()
        try:
            from efficient_kan import KAN
        except ImportError:
            raise ImportError(
                "Standard Spline-KAN requires the efficient-kan library.\n"
                "Please run: pip install git+https://github.com/Blealtan/efficient-kan.git"
            )
        # Instantiate the efficient Spline-KAN
        self.model = KAN(layer_sizes, grid_size=grid_size, spline_order=spline_order)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)