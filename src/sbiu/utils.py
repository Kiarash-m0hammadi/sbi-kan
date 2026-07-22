import torch
import numpy as np

@torch.no_grad()
def get_tuning_curves(sbiu_layer, freqs_to_test: torch.Tensor, base_freq: float = 1.0) -> torch.Tensor:
    """
    Generates frequency tuning curves for an SBIU layer by feeding it pure sine waves.
    This is required to reproduce Figure 2 in the paper.
    
    Args:
        sbiu_layer (nn.Module): An instance of SBIU, InnerKANLayer, or OuterKANLayer.
        freqs_to_test (torch.Tensor): 1D tensor of frequencies to evaluate (e.g., in Hz).
        base_freq (float): The base_freq used during training to scale the inputs correctly.
        
    Returns:
        torch.Tensor: The world probabilities for each evaluated frequency.
    """
    # Create pure time series t
    t = torch.linspace(0, 1, 1000, device=freqs_to_test.device)
    
    tuning_curves = []
    
    # We evaluate one frequency at a time to keep memory usage low
    for freq in freqs_to_test:
        # Generate pure sine wave: x(t) = sin(2 * pi * f * t)
        pure_sine = torch.sin(2 * torch.pi * freq * t).unsqueeze(-1)
        
        # If testing an Inner layer, we need to repeat the feature n times
        if hasattr(sbiu_layer, 'in_features'):
            pure_sine = pure_sine.repeat(1, sbiu_layer.in_features)
            
        # Extract probabilities
        if hasattr(sbiu_layer, 'return_probs'): # Check if it's the KAN wrapper or raw SBIU
            _, probs = sbiu_layer(pure_sine, return_probs=True)
        else:
            _, probs = sbiu_layer(pure_sine, return_probs=True)
            
        # Average probability activation over time for this frequency
        mean_activation = probs.mean(dim=0)
        tuning_curves.append(mean_activation)
        
    return torch.stack(tuning_curves, dim=0)

@torch.no_grad()
def extract_dominant_worlds(outer_probs: torch.Tensor) -> np.ndarray:
    """
    Extracts the dominant world assignments for unsupervised regime clustering.
    Used for the K-Means clustering step described in Section 8.2.3.
    
    Args:
        outer_probs (torch.Tensor): Shape (Batch, N_o, d_out). The probabilities 
                                    extracted from the SBI-KAN outer layer.
    Returns:
        np.ndarray: Integer labels of the most probable world for the first outer channel.
                    Shape (Batch,)
    """
    # Section 8.2.3: "We extract the outer world probabilities P_{q,m}(x_t) 
    # from the first outer channel (q=1)"
    # PyTorch uses 0-indexing, so q=1 is index 0.
    q1_probs = outer_probs[:, 0, :] 
    
    # Get the index of the world with the highest probability (arg max)
    dominant_worlds = torch.argmax(q1_probs, dim=-1).cpu().numpy()
    return dominant_worlds