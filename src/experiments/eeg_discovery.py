import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm

# Ensure root directory is in path for imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from sbiu.unit import SBIU

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --- 1. Dataset Loader ---
class EEGDataset(Dataset):
    def __init__(self, csv_path):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Could not find {csv_path}.")
            
        df = pd.read_csv(csv_path, index_col=0)
        X_raw = df.drop(columns=['y']).values
        y_raw = df['y'].values - 1 
        
        # --- THE FIX: Instance Normalization ---
        # Normalize each 1-second chunk independently so chaotic 
        # seizure amplitudes don't overpower the network's weights.
        X_mean = X_raw.mean(axis=1, keepdims=True)
        X_std = X_raw.std(axis=1, keepdims=True) + 1e-8
        X_raw = (X_raw - X_mean) / X_std
        
        self.X = torch.tensor(X_raw, dtype=torch.float32)
        self.y = torch.tensor(y_raw, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- 2. Real-World SBIU Autoencoder ---
class EEGSpectralAutoencoder(nn.Module):
    def __init__(self, in_features=178, dim=10, init_scale=0.01):
        super().__init__()
        # Real FFT of 178 points yields (178 // 2) + 1 = 90 frequency bins
        self.fft_dim = in_features // 2 + 1 
        
        # 1. Project the 90 frequency bins into the d-dimensional bottleneck
        self.projection = nn.Linear(self.fft_dim, dim)
        
        # 2. SBIU Bottleneck (Creates mutually exclusive regime probabilities)
        self.sbiu = SBIU(dim=dim, shape=())
        nn.init.normal_(self.sbiu.basis.h_params, mean=0.0, std=init_scale)
        
        # 3. Decoder: Tries to reconstruct the FFT from the probability simplex
        self.decoder = nn.Linear(dim, self.fft_dim)
        
    def forward(self, x, return_probs=False):
            # Time domain to Frequency domain
            x_fft = torch.abs(torch.fft.rfft(x, dim=-1))
            
            # --- THE FIX: Log Transform ---
            # Tame the massive spectral peaks so the probability 
            # simplex decoder can actually reconstruct them.
            x_fft = torch.log1p(x_fft)
            
            # Map to Orthogonal Hilbert Space
            state = self.projection(x_fft)
            state = nn.functional.normalize(state, p=2, dim=-1)
            
            # Get World Probabilities
            _, probs = self.sbiu(state, return_probs=True, pre_encoded=True)
            
            # Reconstruct Frequency signature
            recon = self.decoder(probs)
            
            if return_probs:
                return recon, probs, x_fft
            return recon, x_fft

# --- 3. Training and Evaluation Pipeline ---
def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("\n--- Loading EEG Dataset ---")
    csv_path = 'src/data/eeg/Epileptic Seizure Recognition.csv'
    dataset = EEGDataset(csv_path)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    # We set dim=10 (providing a bit of headroom to find the 5 true clinical states)
    model = EEGSpectralAutoencoder(in_features=178, dim=10, init_scale=0.01).to(device)
    
    # Discriminative Learning Rates (Crucial for SBIU stability, as noted in Appendix B)
    basis_params = [p for n, p in model.named_parameters() if 'basis' in n]
    other_params = [p for n, p in model.named_parameters() if 'basis' not in n]
    optimizer = torch.optim.Adam([
        {'params': other_params, 'lr': 0.005, 'weight_decay': 1e-4},
        {'params': basis_params, 'lr': 0.0005, 'weight_decay': 0.0} 
    ])
    
    criterion = nn.MSELoss()
    
    print("\n--- Training Unsupervised Spectral Bottleneck ---")
    model.train()
    epochs = 150
    pbar = tqdm(range(epochs), desc="Training Autoencoder")
    for epoch in pbar:
        total_loss = 0
        for X, _ in dataloader:
            X = X.to(device)
            optimizer.zero_grad()
            
            recon, target_fft = model(X)
            loss = criterion(recon, target_fft)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        pbar.set_postfix({"MSE Loss": f"{total_loss/len(dataloader):.2f}"})
        
    # --- 4. Unsupervised Evaluation (Regime Discovery) ---
    print("\n--- Evaluating Clinical Regimes ---")
    model.eval()
    
    # Load entire dataset for clustering evaluation
    X_all = dataset.X.to(device)
    labels_all = dataset.y.numpy()
    
    with torch.no_grad():
        _, probs, _ = model(X_all, return_probs=True)
        probs = probs.cpu().numpy()
        
    # The SBIU natively groups data into 'worlds' (the index of the max probability)
    predicted_clusters = np.argmax(probs, axis=-1)
    
    # Calculate 5-Class ARI (All states: Seizure, Tumor, Healthy, Eyes Open, Eyes Closed)
    ari_5_class = adjusted_rand_score(labels_all, predicted_clusters)
    print(f"5-Class Unsupervised ARI: {ari_5_class:.4f}")
    
    # Calculate Binary ARI (Seizure [Label 0] vs. All Non-Seizure [Labels 1-4])
    binary_labels_true = (labels_all == 0).astype(int)
    
    # To map the network's 10 worlds to a binary Seizure/Non-Seizure score cleanly, 
    # we can group the generated clusters using K-Means (k=2) on the probability simplex
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    binary_preds = kmeans.fit_predict(probs)
    
    ari_binary = adjusted_rand_score(binary_labels_true, binary_preds)
    print(f"Binary ARI (Seizure vs Non-Seizure): {ari_binary:.4f}")

    print("\nInterpretation:")
    print("If the ARI is significantly above 0.0, the SBIU successfully separated ")
    print("the physical brainwave states purely by routing their frequency geometries ")
    print("through the orthogonal probability simplex—without ever seeing a label!")
    print("\n" + "="*50)
    print("AUDITING THE SBIU WORLDS")
    print("="*50)
    
    import pandas as pd
    
    # Create a DataFrame mapping what the model predicted vs the true label
    df_eval = pd.DataFrame({
        'True_Class': labels_all, 
        'SBIU_World': predicted_clusters
    })
    
    # Map the integers back to what they physically mean in the Kaggle dataset
    class_names = {
        0: "1 - Seizure", 
        1: "2 - Tumor Area", 
        2: "3 - Healthy Area", 
        3: "4 - Eyes Closed", 
        4: "5 - Eyes Open"
    }
    df_eval['True_Class'] = df_eval['True_Class'].map(class_names)
    
    # Generate a cross-tabulation matrix
    crosstab = pd.crosstab(df_eval['SBIU_World'], df_eval['True_Class'])
    print(crosstab)
    print("="*50)
    print("If the SBIU is working, you should see 'Seizures' densely packed into")
    print("just 1 or 2 specific Worlds, physically separated from the rest.")

if __name__ == "__main__":
    main()