from utils.config_loader import ConfigLoader
from ESN import ESN
import os

import logging
logger = logging.getLogger(__name__)


def train(train_data, model):
    pass

def test(test_data, model):
    pass


def run(config):

    train_data, test_data = config.dataset
    model = config.esn.model

    train(train_data, model)
    test(test_data, model)


if __name__ == "__main__":
    exp_path = "./experiments"
    config = ConfigLoader(exp_path)

    logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(levelname)s:%(message)s',
            handlers=[
                logging.FileHandler(os.path.join(exp_path, "run.log")),
                logging.StreamHandler()
            ]
    )

    run(config.conf)