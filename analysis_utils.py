import torch
import numpy as np

def calculate_magnitudes(residuals_list):
    layer_magnitudes = []
    for layer_res in residuals_list:
        magnitudes = torch.linalg.norm(layer_res, dim=-1)
        layer_magnitudes.append(magnitudes.flatten().detach().cpu().numpy())
    
    all_magnitudes = np.concatenate(layer_magnitudes)
    return layer_magnitudes, all_magnitudes
