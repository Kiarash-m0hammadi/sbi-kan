import torch
from kan.inner import InnerKANLayer
from kan.outer import OuterKANLayer
from kan.sbi_kan import SBI_KAN

def test_inner_layer():
    batch_size = 10
    in_features = 6
    out_channels = 13 # 2n + 1
    dim = 4
    
    layer = InnerKANLayer(in_features, out_channels, dim)
    x = torch.randn(batch_size, in_features)
    
    s, probs = layer(x, return_probs=True)
    
    assert s.shape == (batch_size, out_channels), "Inner sum s_q shape mismatch."
    assert probs.shape == (batch_size, out_channels, in_features, dim), "Inner probs shape mismatch."

def test_outer_layer():
    batch_size = 10
    num_channels = 13
    dim = 8
    
    layer = OuterKANLayer(num_channels, dim)
    s = torch.randn(batch_size, num_channels)
    
    Phi, probs = layer(s, return_probs=True)
    
    assert Phi.shape == (batch_size, num_channels), "Outer Phi shape mismatch."
    assert probs.shape == (batch_size, num_channels, dim), "Outer probs shape mismatch."

def test_sbi_kan_full_forward():
    batch_size = 32
    in_features = 6
    out_features = 1 # e.g., regression task
    
    model = SBI_KAN(in_features=in_features, out_features=out_features, dim_in=4, dim_out=8)
    x = torch.randn(batch_size, in_features)
    
    out, probs_dict = model(x, return_probs=True)
    
    # Check final projection output
    assert out.shape == (batch_size, out_features), "Final regression output mismatch."
    
    # Check regime probability extraction keys
    assert "inner" in probs_dict
    assert "outer" in probs_dict

def test_sbi_kan_gradient_flow():
    """
    Verifies that backpropagation doesn't crash through the Lie algebra matrix 
    exponentials and indexing tricks.
    """
    model = SBI_KAN(in_features=4, out_features=2, num_channels=5)
    x = torch.randn(16, 4)
    target = torch.randn(16, 2)
    
    out = model(x)
    loss = torch.nn.functional.mse_loss(out, target)
    loss.backward()
    
    # Check that H generator gradients are populated
    has_grad = False
    for name, param in model.named_parameters():
        if 'basis.h_params' in name:
            assert param.grad is not None, f"Gradient did not reach {name}"
            assert not torch.all(param.grad == 0), f"Gradients are zero for {name}"
            has_grad = True
            
    assert has_grad, "Could not find h_params in the model to test gradients."