from utils.config_loader import ConfigLoader
from utils.gs_plot import plot_gridsearch_results
from ESN import ESN
from metric import nrmse
import os
import logging
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
    configs, param_names = ConfigLoader.generate_grid_search_configs(exp_path)

    logger.info(f"Found {len(configs)} config(s) to run")

    # Collect (param_values_tuple, nrmse) results
    results = []

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
        results.append((tuple(param_values_tuple), float(test_nrmse)))

    # Plot results if any varied parameters exist
    if param_names:
        plot_gridsearch_results(param_names, results)