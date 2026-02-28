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
    Calculates the relative self-influence of each node using the resolvent.

    TRI(j) = |R_jj| / Σ_k |R_jk|

    where R = (zI - W)^{-1} and z = avg in-degree.  This gives the fraction
    of total influence on node j that comes from itself.

    Args:
        W_res (np.ndarray): The reservoir adjacency matrix.

    Returns:
        np.ndarray: A vector where the i-th element is the relative
                    self-influence (TRI) of node i, in [0, 1].
    """
    n_reservoir = W_res.shape[0]
    I = np.identity(n_reservoir)

    z = calculate_avg_degree(W_res)

    try:
        R = np.linalg.inv(z * I - W_res)
        diag_abs = np.abs(np.diag(R))
        row_sums = np.sum(np.abs(R), axis=1)
        return diag_abs / row_sums

    except np.linalg.LinAlgError:
        print("Error: The matrix (zI - W_res) is singular and cannot be inverted.")
        return np.full(n_reservoir, np.nan)

def calculate_avg_degree(W_res):
    """
    Calculates the average number of incoming edges (in-degree) for all nodes in the reservoir.

    Args:
        W_res (np.ndarray): The reservoir adjacency matrix.

    Returns:
        float: The average in-degree of the graph.
    """
    # Count non-zero elements in each column (incoming edges)
    in_degrees = np.count_nonzero(W_res, axis=0)
    return np.mean(in_degrees)
