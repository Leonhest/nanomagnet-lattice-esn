import yaml
import os
from data.NARMA10 import NARMA10


class ConfigLoader():
    def __init__(self, exp_path):
        config_path = self._find_conf(exp_path)

        with open(config_path, "r") as f:
            self.conf = yaml.safe_load(f)

        self._init_matrix()
        self._init_readout()
        self._get_data()
        self._get_args()
        self._init_params()



    def _find_conf(self, exp_path):
        for root, _, files in os.walk(exp_path):
            for file in files:
                if "config.yaml" in file:
                    print(os.path.join(root, file))
                    return os.path.join(root, file)
        
        raise ValueError("Config file was not found in provided experiment folder")

    def _get_data(self):
        name = self.conf["data"]["name"]
        match name:
            case "NARMA":
                self.conf["dataset"] = NARMA10(self.conf["data"])
            case "Lorenz":
                pass
            case _:
                raise ValueError("Dataset not found")