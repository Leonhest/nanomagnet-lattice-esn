import gc
import json
import logging
import os

import numpy as np

from runner.single_run import run, run_res_metrics
from utils.config_loader import ConfigLoader
from utils.gs_plot import plot_decile_statistics, plot_gridsearch_results

logger = logging.getLogger(__name__)


def _setup_logging(exp_path: str) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s:%(message)s",
        handlers=[
            # logging.FileHandler(os.path.join(exp_path, "run.log")),
            logging.StreamHandler()
        ],
    )


def _load_configs(exp_path: str):
    configs, param_names = ConfigLoader.generate_grid_search_configs(exp_path)

    num_runs = configs[0].conf.get("num_runs", 1) if configs else 1
    num_unique_configs = len(configs) // num_runs if num_runs > 0 else len(configs)
    logger.info(
        f"Found {num_unique_configs} unique config(s), running each {num_runs} time(s) = {len(configs)} total runs"
    )

    return configs, param_names


def _detect_modes(configs, param_names):
    # Check if res_metrics mode is enabled
    res_metrics_mode = configs[0].conf.get("res_metrics", False) if configs else False
    if res_metrics_mode:
        logger.info("Running in res_metrics mode - computing kernel_quality, generalization, and memory_capacity")

    # Check if plot_deciles is enabled
    plot_deciles = configs[0].conf.get("plot_deciles", False) if configs else False
    if plot_deciles:
        if len(param_names) > 2:
            logger.warning("plot_deciles only works with <= 2 gridsearch params. Disabling decile plotting.")
            plot_deciles = False
        else:
            logger.info("Decile statistics will be computed and plotted")

    return res_metrics_mode, plot_deciles


def _extract_param_key(conf, param_names):
    param_values_tuple = []
    for path in param_names:
        keys = path.split(".")
        cur = conf
        for k in keys:
            cur = cur[k]
        # if list in base config, current should now be a scalar value for this specific config
        param_values_tuple.append(cur)
    return tuple(param_values_tuple)


def _ensure_key_initialized(
    key,
    score_store,
    *,
    res_metrics_mode,
    plot_deciles,
    config_stats=None,
    config_metrics=None,
    config_decile_stats=None,
):
    if key in score_store:
        return

    score_store[key] = []
    if res_metrics_mode:
        config_metrics[key] = {"kernel_quality": [], "generalization": [], "memory_capacity": []}
        return

    config_stats[key] = {
        "node_means": [],
        "node_variances": [],
        "avg_node_mean": [],
        "avg_node_variance": [],
        "mean_spread": [],
        "avg_total_recurrent_influence": [],
    }
    if plot_deciles:
        config_decile_stats[key] = {
            "avg_node_mean": [[] for _ in range(10)],
            "avg_node_variance": [[] for _ in range(10)],
            "mean_spread": [[] for _ in range(10)],
        }


def _record_nrmse_run(
    key,
    *,
    test_score,
    run_result,
    score_store,
    config_stats,
    plot_deciles,
    config_decile_stats,
):
    # Filter out high NRMSE values
    if test_score >= 0.8:
        return

    score_store[key].append(float(test_score))

    run_stats = dict(run_result)
    run_stats.pop("score", None)
    config_stats[key]["node_means"].append(np.array(run_stats["node_means"]))
    config_stats[key]["node_variances"].append(np.array(run_stats["node_variances"]))
    config_stats[key]["avg_node_mean"].append(float(run_stats["avg_node_mean"]))
    config_stats[key]["avg_node_variance"].append(float(run_stats["avg_node_variance"]))
    config_stats[key]["mean_spread"].append(float(run_stats["mean_spread"]))
    config_stats[key]["avg_total_recurrent_influence"].append(float(run_stats["avg_total_recurrent_influence"]))

    # Collect decile statistics if enabled
    if plot_deciles and config_decile_stats is not None and "decile_stats" in run_result:
        for decile_idx in range(10):
            decile_stat = run_result["decile_stats"][decile_idx]
            config_decile_stats[key]["avg_node_mean"][decile_idx].append(float(decile_stat["avg_node_mean"]))
            config_decile_stats[key]["avg_node_variance"][decile_idx].append(float(decile_stat["avg_node_variance"]))
            config_decile_stats[key]["mean_spread"][decile_idx].append(float(decile_stat["mean_spread"]))


def _record_res_metrics_run(key, *, metrics, config_metrics):
    config_metrics[key]["kernel_quality"].append(float(metrics["kernel_quality"]))
    config_metrics[key]["generalization"].append(float(metrics["generalization"]))
    config_metrics[key]["memory_capacity"].append(float(metrics["memory_capacity"]))


def _cleanup_config(config):
    # Clean up large objects to free memory
    if "dataset" in config.conf:
        del config.conf["dataset"]
    if "esn" in config.conf:
        if "model" in config.conf["esn"]:
            del config.conf["esn"]["model"]
        if "W" in config.conf["esn"]:
            del config.conf["esn"]["W"]
    # Delete the ConfigLoader's large objects
    if hasattr(config, "W_res"):
        del config.W_res
    if hasattr(config, "W_in"):
        del config.W_in
    if hasattr(config, "G_res"):
        del config.G_res
    del config.conf


def _run_all_configs(configs, param_names, exp_path, *, res_metrics_mode, plot_deciles):
    score_store = {}
    config_stats = {} if not res_metrics_mode else None
    config_metrics = {} if res_metrics_mode else None  # store all three metrics separately
    config_decile_stats = {} if plot_deciles and not res_metrics_mode else None

    for i, config in enumerate(configs):
        logger.info(f"Running config {i+1}/{len(configs)}")

        if res_metrics_mode:
            metrics = run_res_metrics(config.conf)
            key = _extract_param_key(config.conf, param_names)
            _ensure_key_initialized(
                key,
                score_store,
                res_metrics_mode=True,
                plot_deciles=False,
                config_metrics=config_metrics,
            )
            _record_res_metrics_run(key, metrics=metrics, config_metrics=config_metrics)
        else:
            run_result = run(config.conf, exp_path)
            test_score = run_result["score"]
            key = _extract_param_key(config.conf, param_names)
            _ensure_key_initialized(
                key,
                score_store,
                res_metrics_mode=False,
                plot_deciles=plot_deciles,
                config_stats=config_stats,
                config_decile_stats=config_decile_stats,
            )
            _record_nrmse_run(
                key,
                test_score=test_score,
                run_result=run_result,
                score_store=score_store,
                config_stats=config_stats,
                plot_deciles=plot_deciles,
                config_decile_stats=config_decile_stats,
            )

        _cleanup_config(config)
        del config

        # Force garbage collection periodically (every 1000 configs or at the end)
        if (i + 1) % 1000 == 0 or (i + 1) == len(configs):
            gc.collect()

    return score_store, config_stats, config_metrics, config_decile_stats


def _aggregate_res_metrics(config_metrics):
    """
    Average kernel_quality, generalization, and memory_capacity across runs.

    Returns:
        Dict[param_key, Dict[str, float]]: summary_stats suitable for plotting (same shape
        as the NRMSE summary_stats dict).
    """
    summary_stats = {}

    for key, metric_store in config_metrics.items():
        kq_values = metric_store["kernel_quality"]
        gen_values = metric_store["generalization"]
        mc_values = metric_store["memory_capacity"]

        avg_kq = np.mean(kq_values)
        avg_gen = np.mean(gen_values)
        avg_mc = np.mean(mc_values)

        std_kq = np.std(kq_values)
        std_gen = np.std(gen_values)
        std_mc = np.std(mc_values)

        logger.info(f"Config {key} - KQ: {avg_kq:.2f}, Gen: {avg_gen:.2f}, MC: {avg_mc:.4f}")

        summary_stats[key] = {
            "kernel_quality": float(avg_kq),
            "kernel_quality_std": float(std_kq),
            "generalization": float(avg_gen),
            "generalization_std": float(std_gen),
            "memory_capacity": float(avg_mc),
            "memory_capacity_std": float(std_mc),
            "n_runs": int(len(mc_values)),
        }

    return summary_stats


def _aggregate_nrmse_results(score_store, config_stats):
    summary_stats = {}

    for key, score_values in score_store.items():
        if not score_values:
            continue

        avg_score = np.mean(score_values)
        std_score = np.std(score_values)
        stats_entry = config_stats[key]
        node_means_avg = np.mean(np.stack(stats_entry["node_means"]), axis=0).tolist()
        node_variances_avg = np.mean(np.stack(stats_entry["node_variances"]), axis=0).tolist()
        avg_node_mean_avg = float(np.mean(stats_entry["avg_node_mean"]))
        avg_node_variance_avg = float(np.mean(stats_entry["avg_node_variance"]))
        mean_spread_avg = float(np.mean(stats_entry["mean_spread"]))
        avg_tri_avg = float(np.mean(stats_entry["avg_total_recurrent_influence"]))

        summary_stats[key] = {
            "score": float(avg_score),
            "score_std": float(std_score),
            "node_means": node_means_avg,
            "node_variances": node_variances_avg,
            "avg_node_mean": avg_node_mean_avg,
            "avg_node_variance": avg_node_variance_avg,
            "mean_spread": mean_spread_avg,
            "avg_total_recurrent_influence": avg_tri_avg,
            "n_runs": len(score_values),
        }
        logger.info(
            f"Config {key} - Avg NRMSE: {avg_score:.6f} ± {std_score:.6f} | "
            f"Mean(avg): {avg_node_mean_avg:.6f}, Var(avg): {avg_node_variance_avg:.6f}, "
            f"Mean spread: {mean_spread_avg:.6f}, TRI(avg): {avg_tri_avg:.6f}"
        )
    return summary_stats


def _write_summary_stats(summary_stats, exp_path):
    if not summary_stats:
        return None

    stats_path = os.path.join(exp_path, "reservoir_stats_summary.json")
    with open(stats_path, "w") as f:
        json.dump({str(key): value for key, value in summary_stats.items()}, f, indent=2)
    logger.info(f"Reservoir statistics summary saved to {stats_path}")
    return stats_path


def _plot_decile_stats(score_store, config_decile_stats, param_names, exp_path):
    if not config_decile_stats:
        return

    # Average decile statistics across runs
    averaged_decile_stats = {}
    for key in config_decile_stats:
        if key not in score_store or not score_store[key]:
            continue
        averaged_decile_stats[key] = {
            "avg_node_mean": [float(np.mean(config_decile_stats[key]["avg_node_mean"][d])) for d in range(10)],
            "avg_node_variance": [
                float(np.mean(config_decile_stats[key]["avg_node_variance"][d])) for d in range(10)
            ],
            "mean_spread": [float(np.mean(config_decile_stats[key]["mean_spread"][d])) for d in range(10)],
        }

    decile_stat_specs = {
        "avg_node_variance": "Average Node Variance",
        "avg_node_mean": "Average Node Mean",
        "mean_spread": "Mean Spread",
    }
    for stat_key, stat_label in decile_stat_specs.items():
        decile_data = {key: averaged_decile_stats[key][stat_key] for key in averaged_decile_stats}
        if decile_data:
            plot_decile_statistics(param_names, decile_data, exp_path, stat_key, stat_label)


def _finalize(
    exp_path,
    param_names,
    *,
    res_metrics_mode,
    plot_deciles,
    score_store,
    config_stats,
    config_metrics,
    config_decile_stats,
):
    """
    Aggregate results, write summaries, and render plots.
    """
    plot_payload = None

    if res_metrics_mode:
        plot_payload = _aggregate_res_metrics(config_metrics)
    else:
        summary_stats = _aggregate_nrmse_results(score_store, config_stats)
        plot_payload = summary_stats
        _write_summary_stats(summary_stats, exp_path)
        if plot_deciles:
            _plot_decile_stats(score_store, config_decile_stats, param_names, exp_path)

    # Plot results if any varied parameters exist
    if param_names:
        plot_gridsearch_results(
            param_names,
            plot_payload,
            exp_path,
            filter_threshold=0.8,
        )


def main(exp_path: str = "./experiments/test/") -> None:
    """
    Entry point for running an experiment folder (config + optional grid search).
    """
    _setup_logging(exp_path)
    configs, param_names = _load_configs(exp_path)
    res_metrics_mode, plot_deciles = _detect_modes(configs, param_names)

    score_store, config_stats, config_metrics, config_decile_stats = _run_all_configs(
        configs,
        param_names,
        exp_path,
        res_metrics_mode=res_metrics_mode,
        plot_deciles=plot_deciles,
    )

    _finalize(
        exp_path,
        param_names,
        res_metrics_mode=res_metrics_mode,
        plot_deciles=plot_deciles,
        score_store=score_store,
        config_stats=config_stats,
        config_metrics=config_metrics,
        config_decile_stats=config_decile_stats,
    )


