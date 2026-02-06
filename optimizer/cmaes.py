import logging

import cma
import networkx as nx
import numpy as np

from matrix import Matrix, save_tile
from optimizer.fitness import evaluate_tile

logger = logging.getLogger(__name__)


def run_cmaes(config, dataset, output_dir):
    """Run CMA-ES optimization to find optimal tile weights.

    The tile topology is fixed (from config). CMA-ES optimizes one weight
    per directed edge.

    Args:
        config: full experiment config dict.
        dataset: NARMA10 dataset object.
        output_dir: directory to write results.

    Returns:
        tuple: (best_tile_G, best_nrmse)
    """
    opt_conf = config["optimization"]
    cmaes_conf = opt_conf["cmaes"]
    W_args = config["esn"]["W_args"]
    esn_conf = config["esn"]
    num_evals = opt_conf.get("num_evals", 1)

    tile_conf = W_args["W_res_args"]["tile"]
    tile_rows, tile_cols = tile_conf["shape"]
    neighborhood = W_args["W_res_args"].get("neighborhood", "Von_Neumann")

    # Build the fixed tile topology
    dummy_matrix = Matrix.__new__(Matrix)
    dummy_matrix.W_res_args = W_args["W_res_args"]
    tile_topo = dummy_matrix.tetragonal(
        [tile_rows, tile_cols], periodic=True, neighborhood=neighborhood
    )

    # Apply directed_edges stochastically using the optimization seed
    rng = np.random.RandomState(opt_conf.get("seed", 42))
    tile_topo = tile_topo.to_directed()

    dir_frac = W_args["W_res_args"]["directed_edges_fraction"]
    dir_weights = W_args["W_res_args"]["directed_edges_weights"]
    undirected_edges = list(dummy_matrix.tetragonal(
        [tile_rows, tile_cols], periodic=True, neighborhood=neighborhood
    ).edges())
    for u, v in undirected_edges:
        if rng.random() < dir_frac:
            del_u, del_v = (u, v) if rng.random() < 0.5 else (v, u)
            if tile_topo.has_edge(del_u, del_v):
                if dir_weights == 0.0:
                    tile_topo.remove_edge(del_u, del_v)
                else:
                    tile_topo.edges[del_u, del_v]["_dir_scaled"] = True

    # Catalog remaining edges as the parameter vector
    edge_list = list(tile_topo.edges())
    num_params = len(edge_list)
    logger.info(f"CMA-ES: {num_params} parameters (edges in tile)")

    # Pre-compute sign pattern using seed so same params give same reservoir
    sign_frac = W_args["W_res_args"]["sign_frac"]
    edge_signs = np.array([
        -1 if rng.random() < sign_frac else 1
        for _ in edge_list
    ])

    # Pre-compute dir scale for edges selected for direction weakening
    edge_dir_scales = np.array([
        dir_weights if tile_topo.edges[u, v].get("_dir_scaled", False) else 1.0
        for u, v in edge_list
    ])

    def params_to_tile(params):
        """Convert a parameter vector to a tile DiGraph."""
        tile_G = nx.DiGraph()
        for node in tile_topo.nodes():
            tile_G.add_node(node)
        for i, (u, v) in enumerate(edge_list):
            weight = params[i] * edge_signs[i] * edge_dir_scales[i]
            tile_G.add_edge(u, v, weight=float(weight))
        return tile_G

    def fitness(params):
        tile_G = params_to_tile(params)
        return evaluate_tile(tile_G, W_args, esn_conf, dataset, num_evals)

    # CMA-ES setup
    sigma0 = cmaes_conf.get("sigma0", 0.3)
    max_gens = cmaes_conf.get("max_generations", 200)
    pop_size = cmaes_conf.get("pop_size", 20)

    x0 = np.random.RandomState(opt_conf.get("seed", 42)).randn(num_params) * 0.5

    opts = {
        "maxiter": max_gens,
        "popsize": pop_size,
        "seed": opt_conf.get("seed", 42),
        "verbose": 1,
    }
    if "bounds" in cmaes_conf:
        opts["bounds"] = cmaes_conf["bounds"]

    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    best_nrmse = float("inf")
    best_params = None

    generation = 0
    while not es.stop():
        candidates = es.ask()
        fitnesses = [fitness(c) for c in candidates]
        es.tell(candidates, fitnesses)
        es.disp()

        gen_best_idx = np.argmin(fitnesses)
        gen_best = fitnesses[gen_best_idx]
        if gen_best < best_nrmse:
            best_nrmse = gen_best
            best_params = candidates[gen_best_idx].copy()

        generation += 1
        logger.info(
            f"Generation {generation}: best_this_gen={gen_best:.6f}, "
            f"overall_best={best_nrmse:.6f}"
        )

    best_tile = params_to_tile(best_params)

    if opt_conf.get("output", {}).get("save_best_tile", True):
        import os
        save_path = os.path.join(output_dir, "best_tile.json")
        save_tile(best_tile, save_path, metadata={
            "method": "cmaes",
            "best_nrmse": best_nrmse,
            "generations": generation,
            "tile_shape": [tile_rows, tile_cols],
            "neighborhood": neighborhood,
        })
        logger.info(f"Best tile saved to {save_path}")

    return best_tile, best_nrmse
