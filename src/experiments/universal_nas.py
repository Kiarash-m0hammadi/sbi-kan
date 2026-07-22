import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import optuna
from sklearn.cluster import KMeans
from sklearn.decomposition import DictionaryLearning
from tqdm import tqdm

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
    def __init__(self, window_size=50, dim=6, init_scale=0.01):
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

def objective(trial, model_name, dataset, device):
    set_seed(42) # Fairness freeze
    
    # 1. Common Search Space
    dim = trial.suggest_categorical("dim", [6, 8, 12, 16, 32, 64, 128])
    epochs = trial.suggest_categorical("epochs", [100, 150, 200])
    batch_size = trial.suggest_categorical("batch_size", [64, 128])
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 2. Model Specific Setup
    if model_name == "SBI-KAN":
        init_scale = trial.suggest_float("init_scale", 1e-4, 1e-1, log=True)
        lr_other = trial.suggest_float("lr_other", 1e-4, 1e-2, log=True)
        lr_basis = trial.suggest_float("lr_basis", 1e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        
        model = WindowedSBIU(window_size=50, dim=dim, init_scale=init_scale).to(device)
        basis_params = [p for n, p in model.named_parameters() if 'basis' in n]
        other_params = [p for n, p in model.named_parameters() if 'basis' not in n]
        optimizer = torch.optim.Adam([
            {'params': other_params, 'lr': lr_other, 'weight_decay': weight_decay},
            {'params': basis_params, 'lr': lr_basis, 'weight_decay': 0.0}
        ])
        
    elif model_name == "Dictionary":
        # Classical model has no epochs/lr
        X_all = torch.stack([dataset[i][0] for i in range(len(dataset))]).numpy()
        labels_all = np.array([dataset[i][2] for i in range(len(dataset))])
        X_fft = np.abs(np.fft.rfft(X_all, axis=-1))
        
        dict_learner = DictionaryLearning(n_components=dim, transform_algorithm='lasso_lars', random_state=42, max_iter=200)
        sparse_codes = dict_learner.fit_transform(X_fft)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        preds_clusters = kmeans.fit_predict(sparse_codes)
        return calculate_ari(labels_all, preds_clusters)
        
    else:
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        
        if model_name == "MLP-raw":
            model = FeatureExtractor(MLPRaw(in_features=26, out_features=1, hidden_dim=dim), model_type="mlp_raw").to(device)
        elif model_name == "MLP-Fourier":
            model = FeatureExtractor(MLPFourier(in_features=26, out_features=1, dim=dim, base_freq=15.0*np.pi), model_type="mlp_fourier").to(device)
        elif model_name == "Spline-KAN":
            model = FeatureExtractor(SplineKAN(layer_sizes=[26, dim, 1]), model_type="spline_kan").to(device)
        elif model_name == "Fourier-KAN":
            model = FeatureExtractor(FourierKAN(layer_sizes=[26, dim, 1], grid_size=8), model_type="fourier_kan").to(device)
            
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 3. Training Loop
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        for X, Y, _ in dataloader:
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            preds = model(X)
            if preds.dim() == 1: preds = preds.unsqueeze(-1)
            loss = criterion(preds, Y)
            loss.backward()
            optimizer.step()

    # 4. Evaluation
    model.eval()
    X_all = torch.stack([dataset[i][0] for i in range(len(dataset))]).to(device)
    labels_all = np.array([dataset[i][2] for i in range(len(dataset))])
    
    if model_name == "SBI-KAN":
        with torch.no_grad():
            _, probs = model(X_all, return_probs=True)
        preds_clusters = torch.argmax(probs.squeeze(), dim=-1).cpu().numpy()
    else:
        features = model.get_features(X_all).cpu().numpy()
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        preds_clusters = kmeans.fit_predict(features)

    ari = calculate_ari(labels_all, preds_clusters)
    if np.isnan(ari): return -1.0
    return ari

def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    t, y, labels = generate_toy_data(n_points=2000, seed=42, freq_multiplier=15.0)
    dataset = SlidingWindowDataset(y, labels, window_size=50, gap=0)
    
    models_to_test = ["SBI-KAN", "Spline-KAN", "Fourier-KAN", "MLP-raw", "MLP-Fourier", "Dictionary"]
    n_trials = 100 # 100 trials per model is enough to find the peak
    
    final_results = {}
    optimal_params = {}
    
    print("=" * 60)
    print("LAUNCHING UNIVERSAL OPTUNA NAS FOR ALL BASELINES")
    print("=" * 60)
    
    for model_name in models_to_test:
        print(f"\nSearching for optimal hyperparameters for: {model_name}")
        study = optuna.create_study(direction="maximize")
        
        pbar = tqdm(range(n_trials), desc=f"{model_name}")
        for _ in pbar:
            study.optimize(lambda trial: objective(trial, model_name, dataset, device), n_trials=1)
            pbar.set_postfix({"Best ARI": f"{study.best_value:.4f}"})
            
        final_results[model_name] = study.best_value
        optimal_params[model_name] = study.best_params

    print("\n" + "="*50)
    print("THE ULTIMATE TABLE 1: FAIR UNIVERSAL NAS RESULTS")
    print("="*50)
    for name, ari in final_results.items():
        print(f"  {name:<25} | {ari:.4f}  (Optimal Dim: {optimal_params[name]['dim']})")
    print("="*50)

if __name__ == "__main__":
    main()