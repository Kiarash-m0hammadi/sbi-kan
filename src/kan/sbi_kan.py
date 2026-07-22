import torch
import torch.nn as nn
from typing import Tuple, Union

from .inner import InnerKANLayer
from .outer import OuterKANLayer

class SBI_KAN(nn.Module):
    """
    Spectral Basis Interpretable Kolmogorov-Arnold Network (SBI-KAN).
    
    Replaces the univariate B-spline edges of a standard KAN with 
    Spectral Basis Interpretable Units (SBIUs). Capable of unsupervised 
    regime discovery via its outer world probabilities.
    
    Corresponds to Section 7 of the paper.
    """
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 num_channels: int = None,
                 dim_in: int = 4, 
                 dim_out: int = 8, 
                 base_freq: float = 1.0):
        """
        Args:
            in_features (int): Dimension of the input vector (n).
            out_features (int): Dimension of the target prediction.
            num_channels (int): Number of outer channels (N_o). 
                                If None, defaults to 2n + 1 (Kolmogorov-Arnold theorem).
            dim_in (int): Dimension of inner SBIUs (d_in).
            dim_out (int): Dimension of outer SBIUs (d_out).
            base_freq (float): Base frequency for the Fourier feature map.
        """
        super().__init__()
        
        # Section 7.1: N_o = 2n + 1
        self.num_channels = num_channels if num_channels is not None else (2 * in_features + 1)
        
        # Inner edges: shape (N_o, n)
        self.inner = InnerKANLayer(in_features=in_features, 
                                   out_channels=self.num_channels, 
                                   dim=dim_in, 
                                   base_freq=base_freq)
        
        # Outer edges: shape (N_o,)
        self.outer = OuterKANLayer(num_channels=self.num_channels, 
                                   dim=dim_out, 
                                   base_freq=base_freq)
        
        # Final linear projection for multivariate regression (Section 7.1)
        self.head = nn.Linear(self.num_channels, out_features)

    def forward(self, x: torch.Tensor, return_probs: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch, in_features)
            return_probs (bool): If True, returns the world probabilities for regime discovery.
            
        Returns:
            out (torch.Tensor): Network predictions.
            probs_dict (dict, optional): Dictionary containing 'inner' and 'outer' probabilities.
        """
        if return_probs:
            s, inner_probs = self.inner(x, return_probs=True)
            Phi, outer_probs = self.outer(s, return_probs=True)
            out = self.head(Phi)
            return out, {"inner": inner_probs, "outer": outer_probs}
        else:
            s = self.inner(x)
            Phi = self.outer(s)
            out = self.head(Phi)
            return out