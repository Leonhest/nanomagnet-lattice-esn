import yaml
import os
from data.NARMA10 import NARMA10
from readout import Ridge
from activation import Tanh
from ESN import ESN

class ConfigLoader():
    def __init__(self, exp_path):
        config_path = self._find_conf(exp_path)

        with open(config_path, "r") as f:
            self.conf = yaml.safe_load(f)

        self._init_w_in()
        self._init_w_res()
        self._init_readout()
        self._init_f()
        self._get_data()
        self._init_esn()



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

    def _init_f(self):
        name = self.conf["esn"]["f"]["name"]
        match name:
            case "tanh":
                self.conf["esn"]["f"] = Tanh(**self.conf["esn"]["f"]["args"])
            case _:
                raise ValueError("F not found")

    def _init_readout(self):
        name = self.conf["esn"]["readout"]["name"]
        match name:
            case "Ridge":
                self.conf["esn"]["readout"] = Ridge(**self.conf["esn"]["readout"]["args"])
            case _:
                raise ValueError("Readout not found")

    def _init_w_in(self):
        pass

    def _init_w_res(self):
        pass

    def _init_esn(self):
        self.conf["esn"]["model"] = ESN(
            W_in=self.conf["esn"]["W_in"],
            W_res=self.conf["esn"]["W_res"],
            spectral_radius=self.conf["esn"]["spectral_radius"],
            f=self.conf["esn"]["f"],
            washout=self.conf["esn"]["washout"],
            readout=self.conf["esn"]["readout"]
        )
