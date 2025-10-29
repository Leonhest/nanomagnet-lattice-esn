from utils.config_loader import ConfigLoader
from ESN import ESN
from metric import nrmse
import os
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger(__name__)


def train(u_train, y_train, model):
    y_pred_train = model.forward(u_train, y_train)
    y_train = y_train[model.washout:]
    nrmse_value = nrmse(y_pred_train, y_train)
    logger.info(f"NRMSE train: {nrmse_value}")
    print(model.X)
    #plt.hist(model.X, bins=100)
    #plt.show()

def test(u_test, y_test, model):
    y_pred_test = model.forward(u_test)
    y_test = y_test[model.washout:]
    nrmse_value = nrmse(y_pred_test, y_test)
    logger.info(f"NRMSE test: {nrmse_value}")

def run(config):
    dataset = config["dataset"]
    model = config["esn"]["model"]
    #print(model.W.W_res)

    train(dataset.u_train, dataset.y_train, model)
    test(dataset.u_test, dataset.y_test, model)


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
    configs = ConfigLoader.generate_grid_search_configs(exp_path)
    
    logger.info(f"Found {len(configs)} config(s) to run")
    
    # Run each config
    for i, config in enumerate(configs):
        logger.info(f"Running config {i+1}/{len(configs)}")
        run(config.conf)