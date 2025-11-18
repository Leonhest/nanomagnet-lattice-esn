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

def calculate_total_recurrent_influence(W_res):
    """
    Calculates the total influence of each node on itself over all possible
    path lengths in the reservoir graph.

    Args:
        W_res (np.ndarray): The reservoir adjacency matrix.

    Returns:
        np.ndarray: A vector where the i-th element is the total recurrent
                    influence (TRI) of node i.
    """
    spectral_radius = np.max(np.abs(np.linalg.eigvals(W_res)))
    if spectral_radius >= 1:
        print(f"Warning: Spectral radius is {spectral_radius:.4f}, which is >= 1.")
        print("The Neumann series may not converge, and results can be unstable.")

    n_reservoir = W_res.shape[0]
    I = np.identity(n_reservoir)

    try:
        S = np.linalg.inv(I - W_res) - I
        
        total_recurrent_influence = np.diag(S)
        
        return total_recurrent_influence

    except np.linalg.LinAlgError:
        print("Error: The matrix (I - W_res) is singular and cannot be inverted.")
        return np.full(n_reservoir, np.nan)