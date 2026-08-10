# Spectral Basis Interpretable Unit (SBIU) & SBI-KAN

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Official PyTorch Implementation for the paper:**  
*Spectral Basis Interpretable Unit: A Learned Orthogonal Projection Module for Interpretable Function Approximation and Unsupervised Regime Discovery*

## 📖 Overview
The **Spectral Basis Interpretable Unit (SBIU)** is a novel, physics-inspired univariate function approximator. It replaces the opaque B-spline edge functions in Kolmogorov-Arnold Networks (KANs) with a learned orthogonal projection in a Fourier-encoded feature space. 

By constraining the network to operate via a classical analog of the **Quantum Measurement Postulate (Born Rule)**, the SBIU naturally decomposes complex signals into a set of mutually exclusive, spectrally interpretable "states."

When embedded into a Kolmogorov-Arnold topology (**SBI-KAN**), the architecture becomes a powerful engine for **unsupervised dynamical regime discovery**. Instead of relying on entangled black-box hidden units, the SBI-KAN forces its representations through a rigid Information Bottleneck, yielding exact, human-readable frequency tuning curves.

---

## 🔬 Core Theoretical Contributions

This repository contains our implementation alongside the theoretical analyses discussed in the paper:

1. **The Kraus Channel Collapse**  
   Recent Quantum-Inspired architectures simulate complex quantum decoherence channels (Kraus operators) to achieve noise robustness. We provide a rigorous mathematical proof that any trace-preserving dephasing channel diagonal to the measurement basis **algebraically collapses into a simple classical projection**. We strip away the redundant quantum overhead to provide the strictly minimal, uniquely identifiable $O(d^2)$ Hermitian quadratic form.
   
2. **The Rigidity of the Orthogonal Frame**  
   We analyze the optimization landscape of parameterizing weights on the Riemannian manifold of the Special Orthogonal group $SO(d)$ via the Lie Algebra ($V = \exp(H)$). We demonstrate how the strict $V V^\top = I$ constraint prevents the model from locally overfitting to high-frequency noise, forcing an **Information Bottleneck** that acts as a powerful regularizer compared to unconstrained Euclidean MLPs.

---

## 📊 Key Results: Occam's Razor in Action

In our Unsupervised Sliding-Window Regime Discovery benchmark, we compared the SBI-KAN against standard KANs and MLPs. To ensure absolute methodological fairness, **every model was independently tuned via a 100-trial Optuna Neural Architecture Search (NAS)** to find its absolute peak performance.

| Model | Adjusted Rand Index (ARI) | Optimal Bottleneck Dimension ($d^*$) |
| :--- | :---: | :---: |
| **SBI-KAN (Ours)** | **0.6857** | **6** |
| Dictionary + K-means | 0.6716 | 8 |
| MLP-raw | 0.6662 | 12 |
| Spline-KAN | 0.6459 | 12 |
| Fourier-KAN | 0.1827 | 128 |
| MLP-Fourier | 0.1593 | 8 |

**The Insight:** While the unconstrained `MLP-raw` required double the dimensions ($d=12$) to achieve comparable segmentation, the SBI-KAN achieved higher performance using only **the strict minimum geometric capacity ($d=6$)** required to host 3 orthogonal physical frequencies. By mathematically constraining the bottleneck to a rigid $SO(6)$ orthogonal frame, the SBI-KAN filters out high-frequency noise without losing expressive power.

---

## 🛠️ Installation & Setup

We provide a fully self-contained Docker environment to ensure 100% reproducibility.

```bash
# 1. Clone the repository
git clone https://github.com/Kiarash-m0hammadi/sbi-kan.git
cd sbi-kan

# 2. Build and start the Docker container
docker compose up -d --build

# 3. Enter the interactive container shell
docker exec -it sbi-kan-dev bash
```
*(The repository is installed in editable mode `pip install -e .` inside the container, meaning any changes you make to the local `src/` files will instantly reflect in the runtime).*

---

## 📥 Data Setup

To run the real-world clinical experiments (EEG Seizure Isolation and ECG Arrhythmia Validation), fetch the datasets directly via the Kaggle CLI:

```bash
# 1. Create data directories in the project root
mkdir -p data/eeg data/ecg

# 2. Download the UCI Epileptic Seizure Recognition Dataset (EEG)
kaggle datasets download -d harunshimanto/epileptic-seizure-recognition -p data/eeg/
unzip data/eeg/epileptic-seizure-recognition.zip -d data/eeg/

# 3. Download the MIT-BIH Arrhythmia Dataset (ECG)
kaggle datasets download -d shayanfazeli/heartbeat -p data/ecg/
unzip data/ecg/heartbeat.zip -d data/ecg/
```

Confirm that your files are organized as follows:
```text
data/
├── eeg/
│   └── "Epileptic Seizure Recognition.csv"
└── ecg/
    ├── mitbih_test.csv
    └── mitbih_train.csv
```

---

## 🚀 Running the Experiments

All execution scripts are located in the `src/experiments/` directory.

### 1. Synthetic Hidden-Regime Oscillator Experiments

* **Absolute-Time Regression & Visualizations (Figures 1, 2 & 3)**  
  Trains an absolute-time SBIU autoencoder and generates the state probability trajectories, frequency tuning curves, and learned orthogonal weight heatmaps.
  ```bash
  python src/experiments/toy_regression.py
  ```

* **Unsupervised Sliding-Window Benchmark (Table 1)**  
  Evaluates the SBI-Bottleneck against Spline-KAN, Fourier-KAN, MLPs, and classical Dictionary Learning on stationary RFFT features.
  ```bash
  python src/experiments/toy_unsupervised.py
  ```

### 2. Real-World Clinical Validations (EEG & ECG)

* **Unsupervised EEG Seizure Discovery ($d=10$, Table 2)**  
  Processes raw EEG spectra from the UCI Epileptic Seizure Recognition dataset to demonstrate autonomic sparsification and seizure isolation (>90% purity).
  ```bash
  python src/experiments/eeg_discovery.py
  ```

* **High-Dimensional EEG Sub-Regime Investigation ($d=12$, Table 3)**  
  Scales the bottleneck capacity to $d=12$ to analyze fine-grained physiological sub-regimes, including the autonomous discovery of ictal theta rhythms (4–6 Hz).
  ```bash
  python src/experiments/eeg_investigation.py
  ```

* **Unsupervised ECG Arrhythmia Isolation ($d=8$, Table 4)**  
  Evaluates the SBI-Bottleneck on $109,446$ heartbeats from the MIT-BIH Arrhythmia Dataset to isolate Ventricular, Fusion, and Supraventricular ectopic beats without labels.
  ```bash
  python src/experiments/ecg_discovery.py
  ```

* **ECG Morphological QRS Audit (Section 6.3.3)**  
  Extracts and audits the physical heartbeat morphologies routed to each pointer state, measuring peak width (QRS widening) and amplitude changes.
  ```bash
  python src/experiments/ecg_properties.py
  ```

### 3. Neural Architecture Search (NAS) & Sensitivity Analysis

* **SBIU Hyperparameter Sensitivity & Importances (Appendix B)**  
  Runs a 100-trial Optuna search specifically for the SBIU bottleneck to evaluate parameter importances (dimension, Lie algebra initialization scale $\sigma$, and discriminative learning rates).
  ```bash
  python src/experiments/toy_nas.py
  ```

* **Universal NAS Search Across All Baselines (Table 1 Peak Tuning)**  
  Runs independent 100-trial Optuna searches for every baseline architecture (MLP-raw, MLP-Fourier, Spline-KAN, Fourier-KAN, Dictionary Learning) to find their peak-performing configurations.
  ```bash
  python src/experiments/universal_nas.py
  ```

---

## 🧪 Unit Testing & Mathematical Verification

The repository contains a `pytest` suite that empirically verifies all core mathematical invariants claimed in the paper:

```bash
pytest -v
```

### Verified Invariants

* **State Normalization ($L_2$ Norm Preservation):** Verifies $\| |\psi(x)\rangle \|_2 = 1.0$ across all input ranges (`test_encoding.py`).
* **Special Orthogonal Manifold ($SO(d)$):** Ensures $V V^\top = \mathbf{I}$, $V^\top V = \mathbf{I}$, and $\det(V) = +1$ via the Lie algebra exponential map `torch.matrix_exp` (`test_pointer_basis.py`).
* **Born-Rule Probability Simplex:** Confirms $\sum_{m=1}^d p_m(x) = 1.0$ for all soft-clustering assignments (`test_unit.py`).
* **Convex Combination Boundedness:** Ensures $\phi(x) \in [\min \lambda_m, \max \lambda_m]$ (`test_unit.py`).
* **Lie Algebra Gradient Flow:** Tests backpropagation through the matrix exponential to guarantee non-vanishing gradients reach skew-symmetric generators $H$ (`test_kan.py`).

---

## 📐 Formal Verification (Lean 4)

The core mathematical claims of the Spectral Basis Interpretable Unit (including **Theorem 1: Kraus Channel Collapse**, Fourier feature map normalization, and output convex bounds) are formally machine-checked in **Lean 4** using Mathlib.

To verify the proofs locally:

```bash
# 1. Install Lean 4 (via elan, if not already installed)
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
source $HOME/.elan/env

# 2. Navigate to the repository root
cd sbi-kan

# 3. Fetch pre-compiled Mathlib cache (prevents 45+ min build time)
lake exe cache get

# 4. Build and verify all formal proofs
lake build

All formal proof scripts are located in `src/math/Math.lean`.

---

## 📁 Repository Structure

```text
.
├── src/
│   ├── math/                 # Lean 4 formal verification (Math.lean, Math/Basic.lean)
│   ├── sbiu/                 # Core math: Fourier encoding, Lie Algebra SO(d), and Born Rule unit
│   ├── kan/                  # KAN topology: Inner/Outer edge aggregators and Baseline models
│   ├── data/                 # Toy oscillator generation and PyTorch Datasets
│   ├── experiments/          # Main execution scripts (Regression, Clinical, NAS)
│   ├── utils/                # Metrics (ARI) and plotting functions
│   └── tests/                # PyTest suite verifying mathematical constraints
├── data/                     # Downloaded and unzipped physiological datasets (Git ignored)
├── Dockerfile                # PyTorch + CUDA environment
├── docker-compose.yml        # Volume mounting and GPU allocation
└── pyproject.toml            # Dependencies (efficient-kan, optuna, etc.)
```

---

## 📝 Citation

If you find this architecture, the Riemannian optimization insights, or the Kraus Collapse proof useful in your research, please cite our work:

```bibtex
@misc{mohammadi_2026_21710382,
  author       = {Mohammadi, Kiarash},
  title        = {Spectral Basis Interpretable Unit: A Learned
                   Orthogonal Projection Module for Interpretable
                   Function Approximation and Unsupervised Regime
                   Discovery
                  },
  month        = jul,
  year         = 2026,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.21710382},
  url          = {https://doi.org/10.5281/zenodo.21710382},
}
}
```
