from utils.config_loader import ConfigLoader
from utils.gs_plot import plot_gridsearch_results
from ESN import ESN
from metric import nrmse
import os
import logging
import numpy as np
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


if __name__ == "__main__":
    exp_path = "./experiments"
    
    logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(levelname)s:%(message)s',
            handlers=[
                logging.FileHandler(os.path.join(exp_path, "run.log")),
                logging.StreamHandler()
            ]
    )

    # Generate all configs (automatically detects arrays and creates grid search)
    # Each config combination will be repeated num_runs times
    configs, param_names = ConfigLoader.generate_grid_search_configs(exp_path)
    
    num_runs = configs[0].conf.get("num_runs", 1) if configs else 1
    num_unique_configs = len(configs) // num_runs if num_runs > 0 else len(configs)
    logger.info(f"Found {num_unique_configs} unique config(s), running each {num_runs} time(s) = {len(configs)} total runs")

    # Collect (param_values_tuple, nrmse) results - group by unique config and average
    config_results = {}
    
    # Run each config
    for i, config in enumerate(configs):
        logger.info(f"Running config {i+1}/{len(configs)}")
        test_nrmse = run(config.conf)

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
        config_results[key].append(float(test_nrmse))
    
    # Average the results for each unique config
    results = []
    for key, nrmse_values in config_results.items():
        avg_nrmse = np.mean(nrmse_values)
        std_nrmse = np.std(nrmse_values)
        logger.info(f"Config {key} - Average NRMSE: {avg_nrmse:.6f} ± {std_nrmse:.6f}")
        results.append((key, float(avg_nrmse)))

    # Plot results if any varied parameters exist
    if param_names:
        plot_gridsearch_results(param_names, results)