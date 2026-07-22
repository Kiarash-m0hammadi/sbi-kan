import torch
import torch.nn as nn

class PointerBasis(nn.Module):
    """
    Learns an orthogonal pointer basis using the Lie algebra-to-group map.
    
    This module maps d(d-1)/2 unconstrained trainable parameters into a 
    skew-symmetric matrix H, and then applies the matrix exponential to 
    produce a special orthogonal matrix V in SO(d). 
    
    Corresponds to Equations 17 and 24 in the paper.
    """
    def __init__(self, dim: int, shape: tuple = ()):
        """
        Args:
            dim (int): The dimensionality of the Hilbert space (d).
            shape (tuple): The batch dimensions for the basis. For a standard
                           layer, this might be (out_features, in_features).
        """
        super().__init__()
        self.dim = dim
        self.shape = shape
        self.num_params = dim * (dim - 1) // 2
        
        # Initialize the independent parameters H for the Lie algebra so(d).
        # Section 9.4: "we initialise H to small random values (e.g., N(0, 0.01^2))"
        # to prevent vanishing gradients in the matrix exponential.
        self.h_params = nn.Parameter(torch.randn(*shape, self.num_params) * 0.01)
        
        # Pre-compute the lower-triangular indices to efficiently construct H
        tril_indices = torch.tril_indices(row=dim, col=dim, offset=-1)
        self.register_buffer('row_idx', tril_indices[0])
        self.register_buffer('col_idx', tril_indices[1])

    def get_V(self) -> torch.Tensor:
        """
        Constructs and returns the orthogonal matrix V.
        
        Returns:
            torch.Tensor: Orthogonal matrices of shape (*shape, dim, dim).
                          The columns of V correspond to the pointer states |\pi_m>.
        """
        # Start with a tensor of zeros: shape (*shape, dim, dim)
        H = torch.zeros(*self.shape, self.dim, self.dim, 
                        device=self.h_params.device, dtype=self.h_params.dtype)
        
        # Fill the strictly lower triangular part with the learnable parameters
        H[..., self.row_idx, self.col_idx] = self.h_params
        
        # Make the matrix strictly skew-symmetric: H = H_lower - H_lower^T
        H = H - H.transpose(-1, -2)
        
        # Matrix exponential maps the skew-symmetric matrix H (in so(d)) 
        # to an orthogonal matrix V (in SO(d)).
        V = torch.matrix_exp(H)
        
        return V

    def forward(self) -> torch.Tensor:
        """
        Returns the orthogonal matrices.
        """
        return self.get_V()

    def extra_repr(self) -> str:
        return f"dim={self.dim}, shape={self.shape}, params_per_unit={self.num_params}"