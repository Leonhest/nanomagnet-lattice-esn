import logging

from metric import kernel_quality, generalization, memory_capacity

from runner.evaluation import train, test, test_closed_loop
from runner.reservoir_stats import compute_reservoir_statistics

logger = logging.getLogger(__name__)


def run(config, exp_path):
    """
    Run a single NRMSE experiment and return score + reservoir statistics.
    """
    dataset = config["dataset"]
    model = config["esn"]["model"]

    _ = train(dataset.u_train, dataset.y_train, model)

    if config["data"].get("closed_loop", False):
        prediction_horizon = config["data"]["prediction_horizon"]
        test_nrmse = test_closed_loop(dataset, model, prediction_horizon)
    else:
        test_nrmse = test(dataset.u_test, dataset.y_test, model)

    stats = compute_reservoir_statistics(model)
    logger.info(f"Average TRI: {stats['avg_tri']:.6f}, Average TRI Ratio: {stats['avg_tri_ratio']:.6f}")

    if config.get("state_plot", False):
        model.plot_node_states(num_nodes=10, save_path=exp_path)

    return {"score": float(test_nrmse), **stats}


def run_res_metrics(config):
    """
    Run reservoir metrics: kernel_quality, generalization, and memory_capacity.
    Returns a dictionary with all three metrics.
    """
    model = config["esn"]["model"]
    _dataset = config["dataset"]  # kept for symmetry with normal run/config requirements

    ks = model.hidden_nodes

    kq = kernel_quality(20, model, ks)
    gen = generalization(20, model, ks)
    mc = memory_capacity(model)

    logger.info(f"Kernel Quality: {kq}, Generalization: {gen}, Memory Capacity: {mc:.4f}")

    # Return all metrics, but use memory_capacity as the main score for plotting
    return {
        "kernel_quality": kq,
        "generalization": gen,
        "memory_capacity": mc,
        "score": mc,
    }


