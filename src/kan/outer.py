import torch
import torch.nn as nn
from typing import Tuple, Union

from sbiu.unit import SBIU

class OuterKANLayer(nn.Module):
    """
    Outer layer of the SBI-KAN architecture.
    
    Applies an independent SBIU to each of the aggregated inner sums s_q(x).
    The outer world probabilities represent full multivariate spectral regimes.
    
    Corresponds to the \Phi_q(s_q) operation in Section 7.
    """
    def __init__(self, num_channels: int, dim: int, base_freq: float = 1.0):
        """
        Args:
            num_channels (int): Number of outer channels (N_o), matching the output of the inner layer.
            dim (int): Dimensionality of the outer Hilbert space (d_out).
            base_freq (float): Base frequency for the outer encodings.
        """
        super().__init__()
        self.num_channels = num_channels
        
        # Instantiate the batched SBIU.
        # Shape (N_o,) creates an independent set of pointer states and lambdas 
        # for every aggregated sum s_q.
        self.sbiu = SBIU(dim=dim, shape=(num_channels,), base_freq=base_freq)

    def forward(self, s: torch.Tensor, return_probs: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            s (torch.Tensor): Inner layer aggregates of shape (batch_size, num_channels)
            return_probs (bool): Whether to return the outer world probabilities (crucial for regime discovery).
            
        Returns:
            Phi (torch.Tensor): Outer outputs \Phi_q(s_q) of shape (batch_size, num_channels)
            probs (torch.Tensor, optional): Outer probabilities of shape (batch_size, num_channels, dim).
                                            These are the P_{q,m}(x) values discussed in Section 7.2.
        """
        # s is already (batch_size, num_channels). The SBIU shape is (num_channels,).
        # This aligns perfectly; no unsqueezing is needed.
        if return_probs:
            Phi, probs = self.sbiu(s, return_probs=True)
            return Phi, probs
        else:
            Phi = self.sbiu(s, return_probs=False)
            return Phi

    def extra_repr(self) -> str:
        return f"num_channels={self.num_channels}"