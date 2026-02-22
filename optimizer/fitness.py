import logging

import numpy as np
import torch

from ESN import ESN
from matrix import Matrix
from readout import Ridge
from activation import Tanh
from runner.evaluation import train, test
from utils.formula import spectral_radius

logger = logging.getLogger(__name__)


def evaluate_tile(tile_G, W_args, esn_conf, dataset, num_evals=1,
                  skip_self_connection=False):
    """Evaluate a tile graph by building an ESN and measuring NRMSE.

    Args:
        tile_G: nx.DiGraph with 'weight' attributes on edges.
        W_args: dict with size, W_in_args, W_res_args (same as config esn.W_args).
        esn_conf: dict with spectral_radius, f (args), washout, readout (args).
        dataset: object with u_train, y_train, u_test, y_test tensors.
        num_evals: number of evaluations to average (different W_in each time).
        skip_self_connection: if True, don't add fixed self-connections (tile
            already contains optimized self-loops).

    Returns:
        float: average NRMSE across evaluations (1.0 penalty for degenerate tiles).
    """
    if tile_G.number_of_edges() == 0:
        return 1.0

    scores = []
    for _ in range(num_evals):
        try:
            W = Matrix.from_tile(tile_G, W_args,
                                 skip_self_connection=skip_self_connection)
            spec_rad = spectral_radius(W.W_res)
            if spec_rad == 0 or not np.isfinite(spec_rad):
                return 1.0

            readout = Ridge(**esn_conf["readout"]["args"])
            f = Tanh(**esn_conf["f"]["args"])

            model = ESN(
                W=W,
                spectral_radius=esn_conf["spectral_radius"],
                f=f,
                washout=esn_conf["washout"],
                readout=readout,
            )

            train(dataset.u_train, dataset.y_train, model)
            u_eval = getattr(dataset, 'u_val', dataset.u_test)
            y_eval = getattr(dataset, 'y_val', dataset.y_test)
            nrmse = test(u_eval, y_eval, model)

            if not np.isfinite(nrmse):
                return 1.0

            scores.append(nrmse)
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
            return 1.0

    return float(np.mean(scores))
