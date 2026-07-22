import os
import sys
import torch
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from ecg_discovery import ECGDataset, ECGSpectralAutoencoder

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csv_path = 'src/data/ecg/mitbih_train.csv'
    dataset = ECGDataset(csv_path)
    
    # Load the trained model weights
    model = ECGSpectralAutoencoder(in_features=187, dim=8).to(device)
    model.load_state_dict(torch.load('src/models/ecg_sbiu.pt', map_location=device))
    print("\n[INFO] Loaded trained model weights from src/models/ecg_sbiu.pt")
    model.eval()
    X_all = dataset.X.to(device)
    labels_all = dataset.y.numpy()
    
    with torch.no_grad():
        _, probs, _ = model(X_all, return_probs=True)
        probs = probs.cpu().numpy()
        
    predicted_clusters = np.argmax(probs, axis=-1)
    raw_signals = dataset.X.numpy()
    
    print("\n" + "="*80)
    print("CLINICAL MORPHOLOGY ANALYSIS OF SBIU ECG WORLDS")
    print("="*80)
    print(f"{'World':<10}{'Primary Class':<25}{'Mean Peak Amp':<18}{'Mean QRS Width (samples)':<25}")
    print("-" * 80)
    
    # We will analyze World 3 (Arrhythmia) vs World 0 (Normal) vs others
    for world in sorted(np.unique(predicted_clusters)):
        world_idx = np.where(predicted_clusters == world)[0]
        if len(world_idx) == 0:
            continue
            
        # Get raw signals assigned to this world
        world_signals = raw_signals[world_idx]
        world_labels = labels_all[world_idx]
        
        # Identify the dominant clinical class in this world
        unique_labels, counts = np.unique(world_labels, return_counts=True)
        dom_label = unique_labels[np.argsort(counts)[::-1][0]]
        class_map = {0: "Normal (N)", 1: "Supraventricular (S)", 2: "Ventricular (V)", 3: "Fusion (F)", 4: "Unclassified (Q)"}
        dom_class_name = class_map[dom_label]
        
        # Clinical Metric 1: Mean Peak Amplitude
        peak_amplitudes = np.max(world_signals, axis=1)
        mean_peak = np.mean(peak_amplitudes)
        
        # Clinical Metric 2: Mean QRS Width 
        # (We measure the width of the peak where the signal is above 30% of its max)
        widths = []
        for sig in world_signals:
            max_val = np.max(sig)
            threshold = 0.3 * max_val
            # Count samples where signal exceeds 30% of peak (R-wave width)
            peak_width = np.sum(sig > threshold)
            widths.append(peak_width)
        mean_width = np.mean(widths)
        
        print(f"{world:<10}{dom_class_name:<25}{mean_peak:<18.4f}{mean_width:<25.2f}")

if __name__ == "__main__":
    main()