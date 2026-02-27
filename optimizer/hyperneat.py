import logging
import os

import neat
import networkx as nx
import numpy as np

from matrix import save_tile
from optimizer.fitness import evaluate_tile

logger = logging.getLogger(__name__)


def _build_substrate_coords(hn_conf):
    """Build substrate node coordinates based on config."""
    substrate = hn_conf.get("substrate", "grid")
    if substrate == "grid":
        rows, cols = hn_conf.get("substrate_shape", [3, 3])
        coords = []
        for r in range(rows):
            for c in range(cols):
                coords.append((r, c))
        return coords
    elif substrate == "custom":
        return [tuple(c) for c in hn_conf["substrate_coords"]]
    else:
        raise ValueError(f"Unknown substrate type: {substrate}")


def cppn_to_tile(net, substrate_coords, threshold=0.2):
    """Convert a CPPN network to a tile DiGraph by querying all coordinate pairs.

    The CPPN takes (x1, y1, x2, y2) and outputs a weight value.
    Edges with abs(weight) below threshold are omitted.
    """
    tile = nx.DiGraph()
    for i, (x1, y1) in enumerate(substrate_coords):
        tile.add_node((x1, y1))

    for i, (x1, y1) in enumerate(substrate_coords):
        for j, (x2, y2) in enumerate(substrate_coords):
            output = net.activate([float(x1), float(y1), float(x2), float(y2)])
            weight = output[0]
            if abs(weight) > threshold:
                tile.add_edge((x1, y1), (x2, y2), weight=float(weight))

    return tile


def run_hyperneat(config, dataset, output_dir):
    """Run HyperNEAT optimization to evolve a CPPN that generates tile weights.

    Args:
        config: full experiment config dict.
        dataset: NARMA10 dataset object.
        output_dir: directory to write results.

    Returns:
        tuple: (best_tile_G, best_nrmse)
    """
    opt_conf = config["optimization"]
    hn_conf = opt_conf["hyperneat"]
    W_args = config["esn"]["W_args"]
    esn_conf = config["esn"]
    num_evals = opt_conf.get("num_evals", 1)
    threshold = hn_conf.get("threshold", 0.2)
    generations = hn_conf.get("generations", 300)

    substrate_coords = _build_substrate_coords(hn_conf)
    logger.info(f"HyperNEAT: substrate has {len(substrate_coords)} nodes")

    # Load neat-python config
    cppn_config_path = hn_conf["cppn_config_file"]
    if not os.path.isabs(cppn_config_path):
        cppn_config_path = os.path.join(output_dir, cppn_config_path)
    neat_config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        cppn_config_path,
    )

    best_nrmse = float("inf")
    best_tile = None

    def eval_genomes(genomes, neat_cfg):
        nonlocal best_nrmse, best_tile

        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, neat_cfg)
            tile_G = cppn_to_tile(net, substrate_coords, threshold)
            nrmse = evaluate_tile(tile_G, W_args, esn_conf, dataset, num_evals)
            genome.fitness = -nrmse  # neat-python maximizes fitness

            if nrmse < best_nrmse:
                best_nrmse = nrmse
                best_tile = tile_G
                logger.info(f"New best NRMSE: {best_nrmse:.6f}")

    population = neat.Population(neat_config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    population.run(eval_genomes, generations)

    if best_tile is not None and opt_conf.get("output", {}).get("save_best_tile", True):
        tile_shape = hn_conf.get("substrate_shape", [3, 3])
        save_path = os.path.join(output_dir, "best_tile.json")
        save_tile(best_tile, save_path, metadata={
            "method": "hyperneat",
            "best_nrmse": best_nrmse,
            "generations": generations,
            "threshold": threshold,
            "substrate": hn_conf.get("substrate", "grid"),
        })
        logger.info(f"Best tile saved to {save_path}")

    return best_tile, best_nrmse
