import os
import numpy as np
import matplotlib.pyplot as plt

# Ensure academic styling for LaTeX inclusion
plt.rcParams.update({
    "font.size": 12,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.labelsize": 14,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

def plot_toy_probabilities(t: np.ndarray, 
                           probs: np.ndarray, 
                           true_regimes: np.ndarray, 
                           save_path: str = "paper/figures/toy_world_probabilities.pdf"):
    """
    Generates Figure 1: World probabilities over time overlaid on true regimes.
    
    Args:
        t: Time array.
        probs: Array of shape (time_steps, num_worlds).
        true_regimes: Array of shape (time_steps,) containing ground truth ints.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4.5)) # Slightly taller to accommodate the legend
    
    # 1. Plot the background regimes
    regime_colors = ['#ffcccc', '#ccffcc', '#ccccff']
    
    # Draw background spans
    start_idx = 0
    for i in range(1, len(true_regimes)):
        if true_regimes[i] != true_regimes[i-1] or i == len(true_regimes) - 1:
            reg = true_regimes[start_idx]
            ax.axvspan(t[start_idx], t[i], color=regime_colors[reg], alpha=0.5, lw=0)
            start_idx = i
            
    # 2. Plot the world probabilities
    num_worlds = probs.shape[1]
    world_colors = plt.cm.tab10(np.linspace(0, 1, num_worlds))
    
    for m in range(num_worlds):
        ax.plot(t, probs[:, m], label=f'World {m+1}', color=world_colors[m], linewidth=1.5)
        
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Probability $p_m(t)$')
    
    # Place the legend BELOW the plot in 2 clean rows of 6.
    # This prevents Matplotlib from squeezing the plot axis.
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=6, frameon=True)
    
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()


def plot_tuning_curves(freqs: np.ndarray, 
                       probs: np.ndarray, 
                       save_path: str = "paper/figures/pointer_tuning_curves.pdf"):
    """
    Generates Figure 2: Frequency tuning curves of the pointer states.
    
    Args:
        freqs: Frequencies tested (x-axis).
        probs: Array of shape (len(freqs), num_worlds).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    
    num_worlds = probs.shape[1]
    world_colors = plt.cm.tab10(np.linspace(0, 1, num_worlds))
    
    # Plot only the worlds that actually have strong activations (dominant worlds)
    max_activations = np.max(probs, axis=0)
    dominant_worlds = np.where(max_activations > 0.1)[0]
    
    for m in dominant_worlds:
        ax.plot(freqs, probs[:, m], label=f'Pointer State {m+1}', color=world_colors[m], linewidth=2)
        # Mark the peak
        peak_idx = np.argmax(probs[:, m])
        ax.plot(freqs[peak_idx], probs[peak_idx, m], marker='o', color=world_colors[m])
        
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Activation Probability')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()


def plot_hypnogram(t: np.ndarray, 
                   true_stages: np.ndarray, 
                   predicted_stages: np.ndarray, 
                   save_path: str = "paper/figures/sleep_hypnogram_comparison.pdf"):
    """
    Generates Figure 3: True vs Discovered Hypnogram for Sleep-EDF.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    
    # Standard Sleep Stages: Wake (0), REM (1), N1 (2), N2 (3), N3 (4)
    stage_labels = ['Wake', 'REM', 'N1', 'N2', 'N3']
    y_ticks = [0, 1, 2, 3, 4]
    
    # True hypnogram
    ax1.step(t, true_stages, where='post', color='black', linewidth=1.5)
    ax1.set_yticks(y_ticks)
    ax1.set_yticklabels(stage_labels)
    ax1.invert_yaxis()
    ax1.set_ylabel('Manual Scoring')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    # Predicted hypnogram
    ax2.step(t, predicted_stages, where='post', color='blue', linewidth=1.5)
    ax2.set_yticks(y_ticks)
    ax2.set_yticklabels(stage_labels)
    ax2.invert_yaxis()
    ax2.set_ylabel('SBI-KAN Dominant World')
    ax2.set_xlabel('Time (Epochs)')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()