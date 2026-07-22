import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import optuna
from sklearn.cluster import KMeans

# Ensure root directory is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.toy_oscillator import generate_toy_data, SlidingWindowDataset
from sbiu.unit import SBIU
from utils.metrics import calculate_ari

# --- Dynamic Windowed SBIU that supports NAS-driven Initialization ---
class NASWindowedSBIU(nn.Module):
    def __init__(self, window_size=50, dim=8, init_scale=0.01):
        super().__init__()
        self.fft_dim = window_size // 2 + 1
        self.projection = nn.Linear(self.fft_dim, dim)
        self.sbiu = SBIU(dim=dim, shape=())
        
        # Dynamically re-initialize h_params with the trial's suggested scale
        nn.init.normal_(self.sbiu.basis.h_params, mean=0.0, std=init_scale)
        
    def forward(self, x, return_probs=False):
        x_fft = torch.abs(torch.fft.rfft(x, dim=-1))
        state = self.projection(x_fft)
        state = nn.functional.normalize(state, p=2, dim=-1)
        return self.sbiu(state, return_probs=return_probs, pre_encoded=True)

def objective(trial):
    # Fix the random seeds inside the trial for fair evaluation
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 1. Suggest Hyperparameters ---
    dim = trial.suggest_categorical("dim", [6, 8, 10, 12, 16])
    init_scale = trial.suggest_float("init_scale", 1e-4, 1e-1, log=True)
    
    # Separate learning rates for the projection/lambdas vs the SO(d) basis
    lr_other = trial.suggest_float("lr_other", 1e-4, 1e-2, log=True)
    lr_basis = trial.suggest_float("lr_basis", 1e-5, 1e-3, log=True)
    
    # Weight decay to prevent the rigid frame from overfitting to noise
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    
    epochs = trial.suggest_categorical("epochs", [100, 150, 200, 300])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    
    # --- 2. Setup Data ---
    t, y, labels = generate_toy_data(n_points=2000, seed=42, freq_multiplier=15.0)
    dataset = SlidingWindowDataset(y, labels, window_size=50, gap=0)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # --- 3. Initialize Model & Discriminative Optimizer ---
    model = NASWindowedSBIU(window_size=50, dim=dim, init_scale=init_scale).to(device)
    
    basis_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 'basis' in name:
            basis_params.append(param)
        else:
            other_params.append(param)
            
    optimizer = torch.optim.Adam([
        {'params': other_params, 'lr': lr_other, 'weight_decay': weight_decay},
        {'params': basis_params, 'lr': lr_basis, 'weight_decay': 0.0} # Do not decay orthogonal generators
    ])
    
    criterion = nn.MSELoss()
    
    # --- 4. Train the Model ---
    model.train()
    for epoch in range(epochs):
        for X, Y, _ in dataloader:
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            preds = model(X)
            if preds.dim() == 1:
                preds = preds.unsqueeze(-1)
            loss = criterion(preds, Y)
            loss.backward()
            optimizer.step()
            
    # --- 5. Evaluate the Clustering Performance ---
    model.eval()
    X_all = torch.stack([dataset[i][0] for i in range(len(dataset))]).to(device)
    labels_all = np.array([dataset[i][2] for i in range(len(dataset))])
    
    with torch.no_grad():
        _, probs = model(X_all, return_probs=True)
    
    preds_clusters = torch.argmax(probs.squeeze(), dim=-1).cpu().numpy()
    ari = calculate_ari(labels_all, preds_clusters)
    
    # Handle possible NaN values gracefully
    if np.isnan(ari):
        return -1.0
        
    return ari

def main():
    print("=== Launching SBI-KAN Optuna Neural Architecture Search ===")
    
    # Suppress verbose Optuna logging so we only see the key results
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study = optuna.create_study(direction="maximize")
    
    # We run 100 trials to thoroughly map out the landscape of the SBIU
    n_trials = 100
    pbar = tqdm(range(n_trials), desc="Optuna NAS Progress")
    
    for i in pbar:
        study.optimize(objective, n_trials=1)
        best_trial = study.best_trial
        pbar.set_postfix({"Best ARI": f"{best_trial.value:.4f} (Trial {best_trial.number})"})
        
    print("\n" + "="*50)
    print("OPTUNA NAS COMPLETED")
    print("="*50)
    print(f"Best Trial Score (ARI): {study.best_value:.4f}")
    print("\nBest Hyperparameters:")
    for key, value in study.best_params.items():
        if isinstance(value, float):
            print(f"  {key:<15}: {value:.6f}")
        else:
            print(f"  {key:<15}: {value}")
            
    # Run Optuna Sensitivity analysis to understand the mathematical properties of the SBIU
    print("\n" + "="*50)
    print("SBI-KAN HYPERPARAMETER SENSITIVITY STUDY")
    print("="*50)
    try:
        import optuna.importance as importance
        import optuna.visualization as vis
        
        # Calculate parameter importances
        param_importances = importance.get_param_importances(study)
        print("Feature Importances on SBI-KAN's Clustering Performance:")
        for param_name, importance_val in param_importances.items():
            print(f"  {param_name:<15}: {importance_val*100:.2f}%")
            
        print("\nInsight:")
        most_important = list(param_importances.keys())[0]
        print(f"  The most critical factor is '{most_important}'. Changing this has the")
        print(f"  greatest impact on the alignment of the learned orthogonal basis.")
    except Exception as e:
        print("Install optuna's visualization module if you want parameter importance printouts.")

if __name__ == "__main__":
    from tqdm import tqdm
    main()