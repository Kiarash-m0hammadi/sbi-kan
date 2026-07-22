import math
import torch
import torch.nn as nn

class FourierEncoding(nn.Module):
    """
    Maps a scalar input to a normalized, real-valued quantum state in a 
    Fourier-encoded Hilbert space.
    
    Corresponds to Equation 9 in the paper, adapted for the real orthogonal 
    group SO(d) by using paired sine/cosine features.
    """
    def __init__(self, dim: int, base_freq: float = 1.0):
        """
        Args:
            dim (int): The dimensionality of the Hilbert space (d). Must be even.
            base_freq (float): The base angular frequency (\omega_0).
        """
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"Dimension 'dim' must be even to accommodate cos/sin pairs. Got {dim}.")
        
        self.dim = dim
        self.base_freq = base_freq
        self.num_pairs = dim // 2
        
        # Frequencies: [1*omega_0, 2*omega_0, ..., (d/2)*omega_0]
        # (Section 4.7: \omega_k = k * \omega_0)
        freqs = torch.arange(1, self.num_pairs + 1, dtype=torch.float32) * self.base_freq
        
        # Register as buffer so it moves to device with the module but isn't a trainable parameter
        self.register_buffer('freqs', freqs)
        
        # Normalization factor to ensure <\psi(x)|\psi(x)> = 1
        # Sum of (cos^2 + sin^2) for d/2 pairs is d/2. 
        # Multiplying by sqrt(2/d) makes the squared L2 norm exactly 1.
        self.normalization_factor = math.sqrt(2.0 / self.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (..., in_features)
            
        Returns:
            torch.Tensor: Encoded quantum state |\psi(x)> of shape (..., in_features, dim).
                          The L2 norm along the last dimension is strictly 1.
        """
        # Expand x to broadcast against frequencies: shape (..., in_features, 1)
        x_expanded = x.unsqueeze(-1)
        
        # Multiply by frequencies: shape (..., in_features, num_pairs)
        angles = x_expanded * self.freqs
        
        # Compute cos and sin components
        cos_features = torch.cos(angles)
        sin_features = torch.sin(angles)
        
        # Concatenate along the state dimension to form the d-dimensional vector
        # Shape: (..., in_features, dim)
        encoded = torch.cat([cos_features, sin_features], dim=-1)
        
        # Apply normalization
        encoded = encoded * self.normalization_factor
        
        return encoded

    def extra_repr(self) -> str:
        """Provides extra information when printing the model."""
        return f"dim={self.dim}, base_freq={self.base_freq}"