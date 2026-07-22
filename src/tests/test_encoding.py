import pytest
import torch
from sbiu.encoding import FourierEncoding

def test_odd_dimension_raises_error():
    with pytest.raises(ValueError):
        FourierEncoding(dim=5)

def test_encoding_shape():
    batch_size = 32
    in_features = 10
    dim = 8
    
    x = torch.randn(batch_size, in_features)
    encoder = FourierEncoding(dim=dim)
    encoded = encoder(x)
    
    assert encoded.shape == (batch_size, in_features, dim), "Output shape mismatch."

def test_l2_norm_is_strictly_one():
    """
    Crucial mathematical test: Ensures <\psi(x)|\psi(x)> = 1 for any input.
    """
    dim = 8
    encoder = FourierEncoding(dim=dim)
    
    x = torch.randn(128, 5) * 100  # Random large/small inputs
    encoded = encoder(x)
    
    # Compute L2 norm along the quantum state dimension
    norms = torch.norm(encoded, p=2, dim=-1)
    
    # Assert all norms are 1.0 within float32 precision
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
        "The Fourier encoding failed to preserve the L2 norm = 1.0"