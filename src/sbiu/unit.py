import torch
import torch.nn as nn
from typing import Tuple, Union

from .encoding import FourierEncoding
from .pointer_basis import PointerBasis

class SBIU(nn.Module):
    """
    Spectral Basis Interpretable Unit (SBIU).
    
    A univariate function approximator that maps a scalar input to a convex 
    combination of probabilities over mutually exclusive "worlds" defined by 
    a learned orthogonal basis in a Fourier-encoded Hilbert space.
    
    Corresponds to Section 4 of the paper.
    """
    def __init__(self, dim: int, shape: tuple = (), base_freq: float = 1.0):
        """
        Args:
            dim (int): Dimensionality of the Hilbert space (d). Must be even.
            shape (tuple): The batch dimensions of the unit. 
                           For a simple 1D->1D mapping, use `()`.
                           For an inner KAN layer (n inputs -> N_o outputs), use `(N_o, n)`.
            base_freq (float): The base frequency (\omega_0) for the Fourier mapping.
        """
        super().__init__()
        self.dim = dim
        self.shape = shape
        
        # 1. Fixed Fourier Feature Map (Section 4.1)
        self.encoding = FourierEncoding(dim=dim, base_freq=base_freq)
        
        # 2. Learned Orthogonal Pointer Basis V (Section 4.2)
        self.basis = PointerBasis(dim=dim, shape=shape)
        
        # 3. Output Coefficients \lambda_m (Section 4.3, Eq. 42)
        # Initialized from a standard normal distribution.
        self.lambdas = nn.Parameter(torch.randn(*shape, dim))

    def forward(self, x: torch.Tensor, return_probs: bool = False, pre_encoded: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the SBIU.
        
        Args:
            x (torch.Tensor): Input tensor. Must be broadcastable to `(*batch_dims, *shape)`.
                              If `pre_encoded=True`, the last dimension must be `dim`.
            return_probs (bool): If True, returns both the output and the world probabilities.
            pre_encoded (bool): If True, bypasses the Fourier encoding (used for Wavelet augmentation).
            
        Returns:
            out (torch.Tensor): The scalar outputs \phi(x).
            probs (torch.Tensor, optional): The world probabilities p_m(x) if requested.
        """
        # Step 1: Encode the input into state |\psi(x)>
        if not pre_encoded:
            psi = self.encoding(x)  # Shape: (*batch_dims, *shape, dim)
        else:
            # If providing custom features (e.g., Wavelets from Section 6), just ensure it is L2 normalized.
            psi = nn.functional.normalize(x, p=2, dim=-1)
            
        # Step 2: Retrieve the orthogonal pointer basis V
        V = self.basis()  # Shape: (*shape, dim, dim)
        
        # Step 3: Project the state onto the pointer states (columns of V)
        # We compute <\pi_m | \psi(x)>. Since V's columns are |\pi_m>, this is equivalent
        # to the vector-matrix multiplication psi @ V.
        # psi shape: (..., dim) -> unsqueeze to (..., 1, dim)
        # V shape: (..., dim, dim)
        # Result: (..., 1, dim) -> squeeze to (..., dim)
        psi_expanded = psi.unsqueeze(-2)
        projections = torch.matmul(psi_expanded, V).squeeze(-2)
        
        # Step 4: World Probabilities via the Born Rule (Eq. 35)
        # Squared modulus (real numbers in this implementation)
        probs = projections ** 2  # Shape: (*batch_dims, *shape, dim)
        
        # Step 5: Final convex combination (Eq. 42)
        # \phi(x) = \sum_m \lambda_m * p_m(x)
        out = torch.sum(self.lambdas * probs, dim=-1)
        
        if return_probs:
            return out, probs
        return out

    def get_tuning_curves(self, t_vals: torch.Tensor) -> torch.Tensor:
        """
        Utility to extract the spectral tuning curves of the pointer states.
        Feeds a set of pure sine waves / time values and extracts the probabilities.
        
        Args:
            t_vals: Time steps to evaluate.
        Returns:
            probs: The world probabilities corresponding to each pointer state.
        """
        with torch.no_grad():
            # Pass through returning probabilities only
            _, probs = self.forward(t_vals, return_probs=True)
        return probs

    def extra_repr(self) -> str:
        return f"dim={self.dim}, shape={self.shape}"