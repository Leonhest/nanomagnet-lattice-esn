from utils.config_loader import ConfigLoader
from utils.gs_plot import plot_gridsearch_results
from ESN import ESN
from metric import nrmse, kernel_quality, generalization, memory_capacity
import os
import logging
import numpy as np
import gc
logger = logging.getLogger(__name__)


def train(u_train, y_train, model):
    y_pred_train = model.forward(u_train, y_train)
    y_train = y_train[model.washout:]
    nrmse_value = nrmse(y_pred_train, y_train)
    logger.info(f"NRMSE train: {nrmse_value}")
    return float(nrmse_value)

def test(u_test, y_test, model):
    y_pred_test = model.forward(u_test)
    y_test = y_test[model.washout:]
    nrmse_value = nrmse(y_pred_test, y_test)
    logger.info(f"NRMSE test: {nrmse_value}")
    return float(nrmse_value)

def run(config):
    dataset = config["dataset"]
    model = config["esn"]["model"]
    train_nrmse = train(dataset.u_train, dataset.y_train, model)
    test_nrmse = test(dataset.u_test, dataset.y_test, model)
    return test_nrmse

def run_res_metrics(config):
    """
    Run reservoir metrics: kernel_quality, generalization, and memory_capacity.
    Returns a dictionary with all three metrics.
    """
    model = config["esn"]["model"]
    dataset = config["dataset"]
    
    ks = model.hidden_nodes  
    
    # Run the three metrics
    kq = kernel_quality(20, model, ks)
    gen = generalization(20, model, ks)
    mc = memory_capacity(model)
   
    
    logger.info(f"Kernel Quality: {kq}, Generalization: {gen}, Memory Capacity: {mc:.4f}")
    
    # Return all metrics, but use memory_capacity as the main score for plotting
    return {
        "kernel_quality": kq,
        "generalization": gen,
        "memory_capacity": mc,
        "score": mc  # Use memory capacity as the main score
    }


if __name__ == "__main__":
    exp_path = "./experiments/res_metrics"
    
    logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(levelname)s:%(message)s',
            handlers=[
                #logging.FileHandler(os.path.join(exp_path, "run.log")),
                logging.StreamHandler()
            ]
    )

    # Generate all configs (automatically detects arrays and creates grid search)
    # Each config combination will be repeated num_runs times
    configs, param_names = ConfigLoader.generate_grid_search_configs(exp_path)
    
    num_runs = configs[0].conf.get("num_runs", 1) if configs else 1
    num_unique_configs = len(configs) // num_runs if num_runs > 0 else len(configs)
    logger.info(f"Found {num_unique_configs} unique config(s), running each {num_runs} time(s) = {len(configs)} total runs")

    # Check if res_metrics mode is enabled
    res_metrics_mode = configs[0].conf.get("res_metrics", False) if configs else False
    if res_metrics_mode:
        logger.info("Running in res_metrics mode - computing kernel_quality, generalization, and memory_capacity")

    # Collect (param_values_tuple, score/metrics) results - group by unique config and average
    config_results = {}
    config_metrics = {}  # For res_metrics mode: store all three metrics separately
    
    # Run each config
    for i, config in enumerate(configs):
        logger.info(f"Running config {i+1}/{len(configs)}")
        
        if res_metrics_mode:
            metrics = run_res_metrics(config.conf)
            test_score = metrics["score"]  # Use memory_capacity as the main score
        else:
            test_score = run(config.conf)
            metrics = None

        # Extract the concrete values for each varied parameter from this config
        param_values_tuple = []
        for path in param_names:
            keys = path.split(".")
            cur = config.conf
            for k in keys:
                cur = cur[k]
            # if list in base config, current should now be a scalar value for this specific config
            param_values_tuple.append(cur)
        
        key = tuple(param_values_tuple)
        if key not in config_results:
            config_results[key] = []
            if res_metrics_mode:
                config_metrics[key] = {"kernel_quality": [], "generalization": [], "memory_capacity": []}
        
        # For res_metrics mode, we don't filter by threshold
        # For normal mode, filter out high NRMSE values
        if not res_metrics_mode:
            if test_score < 0.8:
                config_results[key].append(float(test_score))
        else:
            config_results[key].append(float(test_score))
            # Store all three metrics separately
            config_metrics[key]["kernel_quality"].append(float(metrics["kernel_quality"]))
            config_metrics[key]["generalization"].append(float(metrics["generalization"]))
            config_metrics[key]["memory_capacity"].append(float(metrics["memory_capacity"]))
        
        # Clean up large objects to free memory
        if "dataset" in config.conf:
            del config.conf["dataset"]
        if "esn" in config.conf:
            if "model" in config.conf["esn"]:
                del config.conf["esn"]["model"]
            if "W" in config.conf["esn"]:
                del config.conf["esn"]["W"]
        # Delete the ConfigLoader's large objects
        if hasattr(config, 'W_res'):
            del config.W_res
        if hasattr(config, 'W_in'):
            del config.W_in
        if hasattr(config, 'G_res'):
            del config.G_res
        del config.conf
        del config
        
        # Force garbage collection periodically (every 10 configs or at the end)
        if (i + 1) % 1000 == 0 or (i + 1) == len(configs):
            gc.collect()
    
    # Average the results for each unique config
    results = []
    metrics_results = None  # For res_metrics mode
    metric_name = "Memory Capacity" if res_metrics_mode else "NRMSE"
    
    if res_metrics_mode:
        # Store all three metrics separately
        metrics_results = {"kernel_quality": [], "generalization": [], "memory_capacity": []}
        for key, score_values in config_results.items():
            avg_score = np.mean(score_values)
            std_score = np.std(score_values)
            avg_kq = np.mean(config_metrics[key]["kernel_quality"])
            avg_gen = np.mean(config_metrics[key]["generalization"])
            avg_mc = np.mean(config_metrics[key]["memory_capacity"])
            logger.info(f"Config {key} - KQ: {avg_kq:.2f}, Gen: {avg_gen:.2f}, MC: {avg_mc:.4f}")
            results.append((key, float(avg_score)))
            metrics_results["kernel_quality"].append((key, float(avg_kq)))
            metrics_results["generalization"].append((key, float(avg_gen)))
            metrics_results["memory_capacity"].append((key, float(avg_mc)))
    else:
        for key, score_values in config_results.items():
            avg_score = np.mean(score_values)
            std_score = np.std(score_values)
            logger.info(f"Config {key} - Average {metric_name}: {avg_score:.6f} ± {std_score:.6f}")
            results.append((key, float(avg_score)))

    # Plot results if any varied parameters exist
    if param_names:
        plot_gridsearch_results(param_names, results, exp_path, res_metrics_mode, metrics_results)