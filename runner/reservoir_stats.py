import logging

import numpy as np
import torch

from utils.formula import calculate_resolvent_metrics, orthogonality_error, spectral_radius

logger = logging.getLogger(__name__)


def compute_reservoir_statistics(model, states=None):
    """
    Compute node-level and aggregate reservoir statistics.

    Args:
        model: ESN model instance. Must have `X` (states) available unless
            `states` is provided.
        states: Optional tensor of reservoir states (T, N).
    """
    if states is None:
        if not hasattr(model, "X"):
            raise RuntimeError("Reservoir states not available; run a forward pass first.")
        states = model.X.detach()
    else:
        states = states.detach()

    node_means = torch.mean(states, dim=0)
    node_variances = torch.var(states, dim=0, unbiased=False)
    avg_node_mean = torch.mean(node_means).item()
    avg_node_variance = torch.mean(node_variances).item()
    mean_spread = torch.std(node_means, unbiased=False).item()

    W_res = model.W.W_res.detach().cpu().numpy()
    sr = spectral_radius(W_res)
    if abs(sr - 1.0) < 0.01:
        avg_tri = float('nan')
        avg_tri_ratio = float('nan')
    else:
        tri, tri_ratio = calculate_resolvent_metrics(W_res)
        avg_tri = float(np.mean(tri))
        avg_tri_ratio = float(np.mean(tri_ratio))
    orth_error = orthogonality_error(W_res)

    return {
        "node_means": node_means.cpu().numpy(),
        "node_variances": node_variances.cpu().numpy(),
        "avg_node_mean": avg_node_mean,
        "avg_node_variance": avg_node_variance,
        "mean_spread": mean_spread,
        "avg_tri": avg_tri,
        "avg_tri_ratio": avg_tri_ratio,
        "orthogonality_error": orth_error,
    }


