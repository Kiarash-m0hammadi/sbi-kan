import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import DictionaryLearning

# Ensure root directory is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.toy_oscillator import generate_toy_data, SlidingWindowDataset
from sbiu.unit import SBIU
from kan.baselines import MLPRaw, MLPFourier, FourierKAN, SplineKAN
from utils.metrics import calculate_ari

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --- Wrapper for the Windowed SBIU ---
class WindowedSBIU(nn.Module):
    def __init__(self, window_size=50, dim=6, init_scale=0.011866):
        super().__init__()
        self.fft_dim = window_size // 2 + 1
        self.projection = nn.Linear(self.fft_dim, dim)
        self.sbiu = SBIU(dim=dim, shape=())
        nn.init.normal_(self.sbiu.basis.h_params, mean=0.0, std=init_scale)
        
    def forward(self, x, return_probs=False):
        x_fft = torch.abs(torch.fft.rfft(x, dim=-1))
        state = self.projection(x_fft)
        state = nn.functional.normalize(state, p=2, dim=-1)
        return self.sbiu(state, return_probs=return_probs, pre_encoded=True)

# --- Unified Feature Extractor for Baselines ---
class FeatureExtractor(nn.Module):
    def __init__(self, model, model_type=""):
        super().__init__()
        self.model = model
        self.model_type = model_type
        
    def forward(self, x):
        x_fft = torch.abs(torch.fft.rfft(x, dim=-1))
        return self.model(x_fft)
        
    def get_features(self, x):
        with torch.no_grad():
            x_fft = torch.abs(torch.fft.rfft(x, dim=-1))
            if self.model_type == "mlp_fourier":
                x_flat = self.model.encoding(x_fft).flatten(start_dim=-2)
                for layer in list(self.model.net.children())[:-1]:
                    x_flat = layer(x_flat)
                return x_flat
            elif self.model_type == "mlp_raw":
                x_flat = x_fft
                for layer in list(self.model.net.children())[:-1]:
                    x_flat = layer(x_flat)
                return x_flat
            elif self.model_type == "spline_kan":
                return self.model.model.layers[0](x_fft)
            elif self.model_type == "fourier_kan":
                return self.model.layers[0](x_fft)

def train_and_evaluate(model, dataloader, full_dataset, device, config, model_name=""):
    model.to(device)
    
    if "SBI" in model_name:
        basis_params = []
        other_params = []
        for name, param in model.named_parameters():
            if 'basis' in name:
                basis_params.append(param)
            else:
                other_params.append(param)
        optimizer = torch.optim.Adam([
            {'params': other_params, 'lr': config['lr_other'], 'weight_decay': config['weight_decay']},
            {'params': basis_params, 'lr': config['lr_basis'], 'weight_decay': 0.0}
        ])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config.get('weight_decay', 0.0))
        
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(config['epochs']):
        for X, Y, _ in dataloader:
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            preds = model(X)
            if preds.dim() == 1:
                preds = preds.unsqueeze(-1)
            loss = criterion(preds, Y)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    X_all = torch.stack([full_dataset[i][0] for i in range(len(full_dataset))]).to(device)
    labels_all = np.array([full_dataset[i][2] for i in range(len(full_dataset))])
    
    if "SBI" in model_name:
        with torch.no_grad():
            _, probs = model(X_all, return_probs=True)
        preds_clusters = torch.argmax(probs.squeeze(), dim=-1).cpu().numpy()
    else:
        features = model.get_features(X_all).cpu().numpy()
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        preds_clusters = kmeans.fit_predict(features)

    ari = calculate_ari(labels_all, preds_clusters)
    print(f"{model_name} ARI: {ari:.4f}\n")
    return ari

def run_dictionary_learning(dataset):
    print("Running Dictionary Learning + K-Means...")
    X_all = torch.stack([dataset[i][0] for i in range(len(dataset))]).numpy()
    labels_all = np.array([dataset[i][2] for i in range(len(dataset))])
    
    X_fft = np.abs(np.fft.rfft(X_all, axis=-1))
    # Dictionary learning runs at its own peak of n_components=8
    dict_learner = DictionaryLearning(n_components=8, transform_algorithm='lasso_lars', random_state=42, max_iter=200)
    sparse_codes = dict_learner.fit_transform(X_fft)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    preds_clusters = kmeans.fit_predict(sparse_codes)
    
    ari = calculate_ari(labels_all, preds_clusters)
    print(f"Dictionary Learning ARI: {ari:.4f}\n")
    return ari

def main():
    set_seed(42)
    print("=" * 60)
    print("RUNNING PEAK-TUNED SLIDING WINDOW BENCHMARKS (TABLE 1)")
    print("=" * 60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Generate Dataset (Scaled Frequencies)
    t, y, labels = generate_toy_data(n_points=2000, seed=42, freq_multiplier=15.0)
    
    # 2. Setup Custom Dataloaders for different optimal batch sizes
    dataset_64 = SlidingWindowDataset(y, labels, window_size=50, gap=0)
    dataloader_64 = DataLoader(dataset_64, batch_size=64, shuffle=True)
    
    dataset_128 = SlidingWindowDataset(y, labels, window_size=50, gap=0)
    dataloader_128 = DataLoader(dataset_128, batch_size=128, shuffle=True)
    
    results = {}
    
    # --- MODEL 1: SBI-KAN (Ours) ---
    # Configured with its exact Optuna NAS-discovered optimal hyperparameters (Trial 34, dim=6)
    sbi_config = {
        'lr_other': 0.000132,
        'lr_basis': 0.000635,
        'weight_decay': 0.006778,
        'epochs': 100
    }
    sbi_model = WindowedSBIU(window_size=50, dim=6, init_scale=0.011866)
    results['SBI-KAN (Ours)'] = train_and_evaluate(
        sbiu_model := sbi_model, dataloader_64, dataset_64, device, sbi_config, "SBI-KAN (Ours)"
    )
    
    # --- MODEL 2: Spline-KAN Baseline ---
    # Runs at its own peak of dim=12 with 150 epochs
    spline_config = {'lr': 0.001, 'epochs': 150}
    spline_kan = FeatureExtractor(SplineKAN(layer_sizes=[26, 12, 1]), model_type="spline_kan")
    results['Spline-KAN'] = train_and_evaluate(
        spline_kan, dataloader_128, dataset_128, device, spline_config, "Spline-KAN"
    )
    
    # --- MODEL 3: Fourier-KAN Baseline ---
    # Runs at its own peak of dim=12 with 150 epochs
    fourier_config = {'lr': 0.001, 'epochs': 150}
    fourier_kan = FeatureExtractor(FourierKAN(layer_sizes=[26, 12, 1], grid_size=8), model_type="fourier_kan")
    results['Fourier-KAN'] = train_and_evaluate(
        fourier_kan, dataloader_128, dataset_128, device, fourier_config, "Fourier-KAN"
    )
    
    # --- MODEL 4: MLP Raw Baseline ---
    # Runs at its own peak of dim=12 with 150 epochs
    mlp_raw_config = {'lr': 0.001, 'epochs': 150}
    mlp_raw = FeatureExtractor(MLPRaw(in_features=26, out_features=1, hidden_dim=12), model_type="mlp_raw")
    results['MLP-raw'] = train_and_evaluate(
        mlp_raw, dataloader_128, dataset_128, device, mlp_raw_config, "MLP-raw"
    )
    
    # --- MODEL 5: MLP Fourier Baseline ---
    # Runs at its own peak of dim=12 with 150 epochs
    mlp_four_config = {'lr': 0.001, 'epochs': 150}
    mlp_fourier = FeatureExtractor(MLPFourier(in_features=26, out_features=1, dim=12, base_freq=15.0*np.pi), model_type="mlp_fourier")
    results['MLP-Fourier'] = train_and_evaluate(
        mlp_fourier, dataloader_128, dataset_128, device, mlp_four_config, "MLP-Fourier"
    )
    
    # --- MODEL 6: Dictionary Learning ---
    # Runs at its own peak of n_components=8
    results['Dictionary + K-means'] = run_dictionary_learning(dataset_128)
    
    # Print Table 1 Summary
    print("\n" + "="*50)
    print("Table 1: Windowed ARI Results (Peak-Tuned)")
    print("="*50)
    for name, ari in results.items():
        print(f"  {name:<25} | {ari:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()