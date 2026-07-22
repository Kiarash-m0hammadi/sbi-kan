import torch
import torch.nn as nn
from typing import Tuple, Union

from sbiu.unit import SBIU

class InnerKANLayer(nn.Module):
    """
    Inner layer of the SBI-KAN architecture.
    
    Applies an independent SBIU to each input feature for every outer channel, 
    then sums the results across the input features.
    
    Corresponds to Equation 26 in the paper:
    s_q(x) = \sum_{p=1}^n \phi_{q,p}(x_p)
    """
    def __init__(self, in_features: int, out_channels: int, dim: int, base_freq: float = 1.0):
        """
        Args:
            in_features (int): Number of input features (n).
            out_channels (int): Number of outer channels (N_o, usually 2n+1).
            dim (int): Dimensionality of the inner Hilbert space (d_in).
            base_freq (float): Base frequency (\omega_0) for the inner encodings.
        """
        super().__init__()
        self.in_features = in_features
        self.out_channels = out_channels
        
        # Instantiate the batched SBIU. 
        # Shape (N_o, n) creates an independent set of pointer states and lambdas 
        # for every (outer_channel, input_feature) pair.
        self.sbiu = SBIU(dim=dim, shape=(out_channels, in_features), base_freq=base_freq)

    def forward(self, x: torch.Tensor, return_probs: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features)
            return_probs (bool): Whether to return the inner world probabilities.
            
        Returns:
            s (torch.Tensor): Summed outputs s_q(x) of shape (batch_size, out_channels)
            probs (torch.Tensor, optional): Inner probabilities of shape (batch_size, out_channels, in_features, dim)
        """
        # x is (batch_size, n). We need to broadcast it against the SBIU shape (N_o, n).
        # We unsqueeze to (batch_size, 1, n). PyTorch broadcasting will implicitly 
        # expand the '1' to match 'N_o' during the SBIU's forward pass.
        x_expanded = x.unsqueeze(-2)
        
        if return_probs:
            phi, probs = self.sbiu(x_expanded, return_probs=True)
        else:
            phi = self.sbiu(x_expanded, return_probs=False)
            
        # phi is now (batch_size, out_channels, in_features)
        # Sum across the input features (dim=-1) to compute s_q(x)
        s = torch.sum(phi, dim=-1)
        
        if return_probs:
            return s, probs
        return s

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_channels={self.out_channels}"