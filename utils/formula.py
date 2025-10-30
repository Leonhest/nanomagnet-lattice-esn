import torch
import numpy as np
import logging
logger = logging.getLogger(__name__)    
def spectral_radius(w):
    # Ensure NumPy array input for NumPy eigendecomposition
    if isinstance(w, torch.Tensor):
        w_np = w.detach().cpu().numpy()
    else:
        w_np = np.asarray(w)

    eigvals = np.linalg.eigvals(w_np)
    spectral_radius = float(np.max(np.abs(eigvals)))
    return spectral_radius
