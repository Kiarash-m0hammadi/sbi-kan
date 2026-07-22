import numpy as np
from sklearn.metrics import adjusted_rand_score

def calculate_ari(true_labels: np.ndarray, predicted_clusters: np.ndarray) -> float:
    """
    Computes the Adjusted Rand Index (ARI) to evaluate unsupervised regime discovery.
    
    ARI measures the similarity between two clusterings, correcting for chance.
    A score of 1.0 indicates perfect agreement, while 0.0 indicates random assignment.
    
    Paper Reference: Section 8.1.3 and Table 1.
    """
    return adjusted_rand_score(true_labels, predicted_clusters)

def extract_peak_frequencies(tuning_curves: np.ndarray, test_freqs: np.ndarray) -> np.ndarray:
    """
    Finds the frequency that maximally activates each pointer state (world).
    
    Args:
        tuning_curves (np.ndarray): Shape (num_test_freqs, num_worlds). 
                                    The probabilities for each frequency.
        test_freqs (np.ndarray): Shape (num_test_freqs,). The frequencies evaluated.
        
    Returns:
        np.ndarray: The peak frequency for each world.
        
    Paper Reference: Section 8.1.3 "Spectral interpretability score"
    """
    # Find the index of the maximum probability across the frequency axis (0)
    peak_indices = np.argmax(tuning_curves, axis=0)
    return test_freqs[peak_indices]

def map_clusters_to_truth(true_labels: np.ndarray, predicted_clusters: np.ndarray) -> np.ndarray:
    """
    Maps unsupervised cluster IDs to the ground truth labels they most frequently co-occur with.
    This is purely for visualization/hypnogram matching, not for cheating on metrics (ARI is invariant to label permutation).
    """
    unique_clusters = np.unique(predicted_clusters)
    mapped_clusters = np.zeros_like(predicted_clusters)
    
    for cluster_id in unique_clusters:
        # Get true labels where this cluster was predicted
        mask = (predicted_clusters == cluster_id)
        if np.sum(mask) == 0:
            continue
        # Find the most common true label
        most_frequent_true = np.bincount(true_labels[mask]).argmax()
        mapped_clusters[mask] = most_frequent_true
        
    return mapped_clusters