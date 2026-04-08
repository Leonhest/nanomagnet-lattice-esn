import logging
import os
from multiprocessing import Pool

import cma
import numpy as np

from matrix import Matrix, OffsetTile, save_tile, _resolve_neighborhood_offsets
from optimizer.fitness import evaluate_tile

logger = logging.getLogger(__name__)


class _FitnessEvaluator:
    """Picklable fitness evaluator for multiprocessing."""

    def __init__(self, param_list, edge_signs, edge_dir_scales, tile_shape,
                 neighborhood, W_args, esn_conf, dataset, num_evals,
                 optimize_self_connections):
        self.param_list = param_list
        self.edge_signs = edge_signs
        self.edge_dir_scales = edge_dir_scales
        self.tile_shape = tile_shape
        self.neighborhood = neighborhood
        self.W_args = W_args
        self.esn_conf = esn_conf
        self.dataset = dataset
        self.num_evals = num_evals
        self.optimize_self_connections = optimize_self_connections

    def params_to_tile(self, params):
        offset_weights = {}
        for i, (pos, offset) in enumerate(self.param_list):
            weight = params[i] * self.edge_signs[i] * self.edge_dir_scales[i]
            offset_weights[(pos, offset)] = float(weight)
        return OffsetTile(self.tile_shape, offset_weights, self.neighborhood)

    def __call__(self, params):
        tile = self.params_to_tile(params)
        return evaluate_tile(
            tile, self.W_args, self.esn_conf, self.dataset, self.num_evals,
            skip_self_connection=self.optimize_self_connections,
        )


def run_cmaes(config, dataset, output_dir):
    """Run CMA-ES optimization to find optimal tile weights.

    Optimizes one weight per (tile_position, offset) pair, which avoids
    aliasing on small tiles where multiple offsets map to the same (src, dst).

    Args:
        config: full experiment config dict.
        dataset: NARMA10 dataset object.
        output_dir: directory to write results.

    Returns:
        tuple: (best_tile, best_nrmse) where best_tile is an OffsetTile.
    """
    opt_conf = config["optimization"]
    cmaes_conf = opt_conf["cmaes"]
    W_args = config["esn"]["W_args"]
    esn_conf = config["esn"]
    num_evals = opt_conf.get("num_evals", 1)

    tile_conf = W_args["W_res_args"]["tile"]
    tile_rows, tile_cols = tile_conf["shape"]
    tile_shape = (tile_rows, tile_cols)
    neighborhood = W_args["W_res_args"].get("neighborhood", "Von_Neumann")

    optimize_signs = cmaes_conf.get("optimize_signs", False)
    optimize_directions = cmaes_conf.get("optimize_directions", False)
    optimize_self_connections = cmaes_conf.get("optimize_self_connections", False)

    rng = np.random.RandomState(opt_conf.get("seed", 42))

    # Build parameter list: one entry per (tile_position, offset)
    offsets = _resolve_neighborhood_offsets(neighborhood)
    tile_nodes = [(r, c) for r in range(tile_rows) for c in range(tile_cols)]

    param_list = []  # [(tile_pos, (dr, dc)), ...]
    for pos in tile_nodes:
        for offset in offsets:
            param_list.append((pos, offset))

    if optimize_self_connections:
        for pos in tile_nodes:
            param_list.append((pos, (0, 0)))

    num_params = len(param_list)
    logger.info(f"CMA-ES: {num_params} parameters ({len(tile_nodes)} positions × {len(offsets)} offsets"
                f"{' + self-connections' if optimize_self_connections else ''})")

    # Pre-compute sign pattern per (pos, offset)
    if not optimize_signs:
        sign_frac = W_args["W_res_args"]["sign_frac"]
        edge_signs = np.array([
            1 if offset == (0, 0) else (-1 if rng.random() < sign_frac else 1)
            for _, offset in param_list
        ])
    else:
        edge_signs = np.ones(num_params)

    # Pre-compute direction scaling per (pos, offset)
    if not optimize_directions:
        dir_frac = W_args["W_res_args"]["directed_edges_fraction"]
        dir_weight_scale = W_args["W_res_args"]["directed_edges_weights"]
        edge_dir_scales = np.ones(num_params)
        processed = set()
        for i, (pos, (dr, dc)) in enumerate(param_list):
            if (dr, dc) == (0, 0):
                continue
            # Find the reverse: the param at the offset-destination going back
            reverse_pos = ((pos[0] + dr) % tile_rows, (pos[1] + dc) % tile_cols)
            reverse_offset = (-dr, -dc)
            pair_key = tuple(sorted([(pos, (dr, dc)), (reverse_pos, reverse_offset)]))
            if pair_key in processed:
                continue
            processed.add(pair_key)
            if rng.random() < dir_frac:
                # Find the reverse param index
                try:
                    j = param_list.index((reverse_pos, reverse_offset))
                except ValueError:
                    continue
                # Randomly pick which direction to scale
                if rng.random() < 0.5:
                    edge_dir_scales[i] = dir_weight_scale
                else:
                    edge_dir_scales[j] = dir_weight_scale
    else:
        edge_dir_scales = np.ones(num_params)

    evaluator = _FitnessEvaluator(
        param_list, edge_signs, edge_dir_scales, tile_shape,
        neighborhood, W_args, esn_conf, dataset, num_evals,
        optimize_self_connections,
    )

    n_workers = int(os.environ.get("ESN_WORKERS", os.cpu_count() or 1))

    # CMA-ES setup
    sigma0 = cmaes_conf.get("sigma0", 0.3)
    max_gens = cmaes_conf.get("max_generations", 200)
    pop_size = cmaes_conf.get("pop_size", 20)

    init_rng = np.random.RandomState(opt_conf.get("seed", 42))
    if "bounds" in cmaes_conf:
        lo, hi = cmaes_conf["bounds"]
        x0 = init_rng.uniform(lo, hi, size=num_params)
    else:
        x0 = init_rng.randn(num_params) * 0.5

    opts = {
        "maxiter": max_gens,
        "popsize": pop_size,
        "seed": opt_conf.get("seed", 42),
        "verbose": 1,
    }
    if "bounds" in cmaes_conf:
        opts["bounds"] = cmaes_conf["bounds"]

    # Covariance mode: "full" (default), "diagonal" (sep-CMA-ES),
    # or an integer N to use diagonal for the first N iterations then full.
    cov_mode = cmaes_conf.get("covariance", "full")
    if cov_mode == "diagonal":
        opts["CMA_diagonal"] = True
    elif cov_mode != "full":
        opts["CMA_diagonal"] = int(cov_mode)

    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    best_nrmse = float("inf")
    best_params = None
    history = {"gen_best": [], "overall_best": [], "gen_mean": []}

    # Prevent BLAS thread contention across workers
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

    generation = 0
    logger.info(f"CMA-ES: using {n_workers} parallel workers")
    pool = Pool(n_workers) if n_workers > 1 else None

    while not es.stop():
        candidates = es.ask()
        if pool is not None:
            fitnesses = pool.map(evaluator, candidates)
        else:
            fitnesses = [evaluator(c) for c in candidates]
        es.tell(candidates, fitnesses)
        es.disp()

        gen_best_idx = np.argmin(fitnesses)
        gen_best = fitnesses[gen_best_idx]
        if gen_best < best_nrmse:
            best_nrmse = gen_best
            best_params = candidates[gen_best_idx].copy()

        generation += 1
        history["gen_best"].append(gen_best)
        history["overall_best"].append(best_nrmse)
        history["gen_mean"].append(float(np.mean(fitnesses)))
        logger.info(
            f"Generation {generation}: best_this_gen={gen_best:.6f}, "
            f"overall_best={best_nrmse:.6f}"
        )

    if pool is not None:
        pool.close()
        pool.join()

    best_tile = evaluator.params_to_tile(best_params)

    if opt_conf.get("output", {}).get("save_best_tile", True):
        save_path = os.path.join(output_dir, "best_tile.json")
        save_tile(best_tile, save_path, metadata={
            "method": "cmaes",
            "best_nrmse": best_nrmse,
            "generations": generation,
            "neighborhood": neighborhood,
            "optimize_signs": optimize_signs,
            "optimize_directions": optimize_directions,
            "optimize_self_connections": optimize_self_connections,
        })
        logger.info(f"Best tile saved to {save_path}")

    return best_tile, best_nrmse, history
