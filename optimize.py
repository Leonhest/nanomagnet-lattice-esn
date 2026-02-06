import logging
import os
import sys

import yaml

from data.NARMA10 import NARMA10

DEFAULT_EXP_PATH = "./experiments/optimize_test/"


def _setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s:%(message)s",
        handlers=[logging.StreamHandler()],
    )


def _find_config(exp_path):
    for root, _, files in os.walk(exp_path):
        for file in files:
            if "config.yaml" in file:
                return os.path.join(root, file)
    raise ValueError("Config file was not found in provided experiment folder")


def main(exp_path):
    _setup_logging()
    logger = logging.getLogger(__name__)

    config_path = _find_config(exp_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    opt_conf = config.get("optimization", {})
    method = opt_conf.get("method", "static")

    if method == "static":
        logger.info("Optimization method is 'static'. Nothing to optimize. Use run.py instead.")
        return

    # Load dataset once
    logger.info("Loading dataset...")
    dataset = NARMA10(config["data"])

    os.makedirs(exp_path, exist_ok=True)

    if method == "cmaes":
        from optimizer.cmaes import run_cmaes

        best_tile, best_nrmse = run_cmaes(config, dataset, exp_path)
        logger.info(f"CMA-ES finished. Best NRMSE: {best_nrmse:.6f}")

    elif method == "hyperneat":
        from optimizer.hyperneat import run_hyperneat

        best_tile, best_nrmse = run_hyperneat(config, dataset, exp_path)
        logger.info(f"HyperNEAT finished. Best NRMSE: {best_nrmse:.6f}")

    else:
        raise ValueError(f"Unknown optimization method: {method}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        DEFAULT_EXP_PATH = sys.argv[1]
    main(DEFAULT_EXP_PATH)
