import logging

import numpy as np
import torch

from utils.formula import calculate_total_recurrent_influence

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
    total_recurrent_influence = calculate_total_recurrent_influence(W_res)
    avg_total_recurrent_influence = float(np.mean(total_recurrent_influence))

    return {
        "node_means": node_means.cpu().numpy(),
        "node_variances": node_variances.cpu().numpy(),
        "avg_node_mean": avg_node_mean,
        "avg_node_variance": avg_node_variance,
        "mean_spread": mean_spread,
        "avg_total_recurrent_influence": avg_total_recurrent_influence,
    }


def compute_decile_statistics(model, num_deciles=10):
    """
    Split the reservoir states into `num_deciles` chunks over time and compute
    statistics for each chunk.
    """
    if not hasattr(model, "X"):
        raise RuntimeError("Reservoir states not available for decile computation.")

    states = model.X.detach()
    num_timesteps = states.shape[0]
    decile_size = num_timesteps // num_deciles

    decile_stats = []
    for decile_idx in range(num_deciles):
        start_idx = decile_idx * decile_size
        end_idx = (decile_idx + 1) * decile_size if decile_idx < (num_deciles - 1) else num_timesteps
        decile_states = states[start_idx:end_idx]
        decile_stats.append(compute_reservoir_statistics(model, states=decile_states))

    return decile_stats


