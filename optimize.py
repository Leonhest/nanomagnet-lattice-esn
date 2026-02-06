import copy
import logging
import os
import sys
from itertools import product

import yaml

from data.NARMA10 import NARMA10
from matrix import save_tile
from utils.config_loader import ConfigLoader

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


def _generate_grid_configs(base_config):
    list_params = ConfigLoader._find_list_parameters(base_config)
    if not list_params:
        return [base_config], []

    param_names = list(list_params.keys())
    param_values = list(list_params.values())
    combinations = list(product(*param_values))

    configs = []
    for combo in combinations:
        config = copy.deepcopy(base_config)
        for param_name, param_value in zip(param_names, combo):
            ConfigLoader._set_nested_value(config, param_name, param_value)
        configs.append(config)

    return configs, param_names


def _extract_param_values(config, param_names):
    values = []
    for name in param_names:
        cur = config
        for k in name.split("."):
            cur = cur[k]
        values.append(cur)
    return values


def _make_tile_filename(method, param_names, config):
    parts = [f"best_tile_{method}"]
    for name in param_names:
        keys = name.split(".")
        cur = config
        for k in keys:
            cur = cur[k]
        short_name = keys[-1]
        parts.append(f"{short_name}={cur}")
    return "_".join(parts) + ".json"


def _run_optimization(method, config, dataset, exp_path):
    # Disable internal tile saving — we handle it ourselves
    config.setdefault("optimization", {}).setdefault("output", {})["save_best_tile"] = False

    if method == "cmaes":
        from optimizer.cmaes import run_cmaes

        return run_cmaes(config, dataset, exp_path)
    elif method == "hyperneat":
        from optimizer.hyperneat import run_hyperneat

        return run_hyperneat(config, dataset, exp_path)
    else:
        raise ValueError(f"Unknown optimization method: {method}")


def main(exp_path):
    _setup_logging()
    logger = logging.getLogger(__name__)

    config_path = _find_config(exp_path)
    with open(config_path, "r") as f:
        base_config = yaml.safe_load(f)

    opt_conf = base_config.get("optimization", {})
    method = opt_conf.get("method", "static")

    if method == "static":
        logger.info("Optimization method is 'static'. Nothing to optimize. Use run.py instead.")
        return

    configs, param_names = _generate_grid_configs(base_config)
    logger.info(f"Grid search: {len(configs)} config(s)")

    logger.info("Loading dataset...")
    dataset = NARMA10(base_config["data"])

    os.makedirs(exp_path, exist_ok=True)

    for i, config in enumerate(configs):
        param_values = _extract_param_values(config, param_names)
        param_desc = ", ".join(
            f"{n.split('.')[-1]}={v}" for n, v in zip(param_names, param_values)
        ) if param_names else "default"

        logger.info(f"Config {i+1}/{len(configs)} ({param_desc})")

        best_tile, best_nrmse = _run_optimization(method, config, dataset, exp_path)
        logger.info(f"Finished. Best NRMSE: {best_nrmse:.6f}")

        if best_tile is not None:
            filename = _make_tile_filename(method, param_names, config)
            save_path = os.path.join(exp_path, filename)
            save_tile(best_tile, save_path, metadata={
                "method": method,
                "best_nrmse": best_nrmse,
                "params": dict(zip(param_names, param_values)) if param_names else {},
            })
            logger.info(f"Tile saved to {save_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        DEFAULT_EXP_PATH = sys.argv[1]
    main(DEFAULT_EXP_PATH)
