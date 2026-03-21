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

def calculate_resolvent_metrics(W_res):
    """
    Computes resolvent-based metrics for each node.

    R = (I - W)^{-1}

    TRI(j)       = R_jj              (diagonal of the resolvent)
    TRI_ratio(j) = |R_jj| / Σ_k |R_jk|  (relative self-influence)

    Args:
        W_res (np.ndarray): The reservoir adjacency matrix.

    Returns:
        tri: np.ndarray — diagonal of the resolvent per node.
        tri_ratio: np.ndarray — relative self-influence per node, in [0, 1].
    """
    n_reservoir = W_res.shape[0]
    I = np.identity(n_reservoir)

    try:
        R = np.linalg.inv(I - W_res)
        tri = np.diag(R)
        diag_abs = np.abs(tri)
        row_sums = np.sum(np.abs(R), axis=1)
        tri_ratio = diag_abs / row_sums
        return tri, tri_ratio

    except np.linalg.LinAlgError:
        print("Error: The matrix (I - W_res) is singular and cannot be inverted.")
        return np.full(n_reservoir, np.nan), np.full(n_reservoir, np.nan)

def orthogonality_error(W_res):
    """Frobenius norm of W^T W - I. Zero means perfectly orthogonal."""
    if isinstance(W_res, torch.Tensor):
        W_res = W_res.detach().cpu().numpy()
    else:
        W_res = np.asarray(W_res)
    WtW = W_res.T @ W_res
    return float(np.linalg.norm(WtW - np.eye(WtW.shape[0]), 'fro'))


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
