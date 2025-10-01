from utils.config_loader import ConfigLoader
from ESN import ESN
import os

import logging
logger = logging.getLogger(__name__)


def train(train_data, model, params):
    pass

def test(test_data, model, params):
    pass


def run(config):

    train_data, test_data = config.data
    params = config.params
    model = ESN(params)

    train(train_data, model, params)
    test(test_data, model, params)


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