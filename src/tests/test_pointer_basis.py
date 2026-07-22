import torch
from sbiu.pointer_basis import PointerBasis

def test_pointer_basis_shape():
    dim = 6
    shape = (4, 3) # e.g., N_o=4, n=3
    basis = PointerBasis(dim=dim, shape=shape)
    
    V = basis()
    assert V.shape == (*shape, dim, dim), "Output V shape mismatch."

def test_orthogonality():
    """
    Crucial mathematical test: Ensures V @ V^T = I and V^T @ V = I
    This verifies the matrix exponential maps the Lie algebra to SO(d).
    """
    dim = 4
    shape = (5, 2)
    basis = PointerBasis(dim=dim, shape=shape)
    
    V = basis()
    
    # Create the identity matrix broadcasted to the batch shape
    I = torch.eye(dim).expand(*shape, dim, dim)
    
    # Check V @ V^T = I
    V_Vt = torch.matmul(V, V.mT)
    assert torch.allclose(V_Vt, I, atol=1e-5), "V @ V^T is not the Identity matrix."
    
    # Check V^T @ V = I
    Vt_V = torch.matmul(V.mT, V)
    assert torch.allclose(Vt_V, I, atol=1e-5), "V^T @ V is not the Identity matrix."

def test_special_orthogonal_determinant():
    """
    Checks that det(V) = 1 (Special Orthogonal group).
    """
    dim = 4
    basis = PointerBasis(dim=dim, shape=(2,))
    V = basis()
    
    dets = torch.linalg.det(V)
    assert torch.allclose(dets, torch.ones_like(dets), atol=1e-4), "Determinant is not 1."