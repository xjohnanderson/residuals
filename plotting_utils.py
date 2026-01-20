import matplotlib.pyplot as plt
import numpy as np

def plot_distributions(attn_mags, mlp_mags):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(attn_mags, bins=50, color='skyblue', edgecolor='black', alpha=0.7, label='Attention')
    plt.title('Attention Residual Magnitudes')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(mlp_mags, bins=50, color='lightcoral', edgecolor='black', alpha=0.7, label='MLP')
    plt.title('MLP Residual Magnitudes')
    plt.show()

def plot_layer_evolution(attn_layer_mags, mlp_layer_mags):
    plt.figure(figsize=(10, 6))
    plt.plot([np.mean(m) for m in attn_layer_mags], marker='o', label='Attention')
    plt.plot([np.mean(m) for m in mlp_layer_mags], marker='x', label='MLP')
    plt.title('Average Residual Magnitude Across Layers')
    plt.xlabel('Layer Index')
    plt.ylabel('L2 Norm')
    plt.legend()
    plt.grid(True)
    plt.show()
