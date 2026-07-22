import torch
from sbiu.unit import SBIU

def test_sbiu_output_shapes():
    batch_size = 16
    in_features = 5
    out_channels = 3
    dim = 8
    
    # Example input shaped as it would be inside an inner KAN layer
    x = torch.randn(batch_size, 1, in_features) 
    
    sbiu = SBIU(dim=dim, shape=(out_channels, in_features))
    out, probs = sbiu(x, return_probs=True)
    
    assert out.shape == (batch_size, out_channels, in_features)
    assert probs.shape == (batch_size, out_channels, in_features, dim)

def test_probabilities_sum_to_one():
    """
    Crucial mathematical test: Ensures \sum p_m(x) = 1.0 (Born rule completeness).
    """
    sbiu = SBIU(dim=4, shape=(3,))
    x = torch.randn(10, 3)  # Batch=10, channels=3
    
    _, probs = sbiu(x, return_probs=True)
    
    prob_sums = torch.sum(probs, dim=-1)
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5), \
        "World probabilities do not sum to 1.0"

def test_output_bounds():
    """
    Checks that the output \phi(x) is strictly bounded by the min and max 
    of the coefficients \lambda_m.
    """
    sbiu = SBIU(dim=8, shape=(2,))
    x = torch.randn(100, 2)
    
    out = sbiu(x)
    
    lambdas = sbiu.lambdas # shape (2, 8)
    min_vals, _ = torch.min(lambdas, dim=-1) # shape (2,)
    max_vals, _ = torch.max(lambdas, dim=-1) # shape (2,)
    
    # Check bounds for each feature
    assert torch.all(out >= min_vals.unsqueeze(0) - 1e-5), "Output fell below minimum lambda."
    assert torch.all(out <= max_vals.unsqueeze(0) + 1e-5), "Output exceeded maximum lambda."