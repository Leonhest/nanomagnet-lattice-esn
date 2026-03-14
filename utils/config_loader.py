from re import L
import yaml
import os
import copy
from itertools import product
from data.NARMA10 import NARMA10
from data.mackey_glass import MackeyGlass
from readout import Ridge
from activation import Hysteresis
from ESN import ESN
from matrix import Matrix

class ConfigLoader():
    def __init__(self, exp_path, config_dict=None):
        if config_dict is None:
            config_path = self._find_conf(exp_path)
            with open(config_path, "r") as f:
                self.conf = yaml.safe_load(f)
        else:
            self.conf = config_dict

        self._init_W()
        self._init_readout()
        self._init_f()
        self._get_data()
        self._init_esn()
    
    @staticmethod
    def generate_grid_search_configs(exp_path):
        config_path = ConfigLoader._find_conf_static(exp_path)
        with open(config_path, "r") as f:
            base_config = yaml.safe_load(f)

        ConfigLoader._expand_folder_paths(base_config)

        # Get num_runs from config
        num_runs = base_config.get("num_runs", 1)
        
        # Find all list parameters
        list_params = ConfigLoader._find_list_parameters(base_config)
        
        if not list_params:
            # Generate num_runs copies of the single config
            configs = [ConfigLoader(exp_path, copy.deepcopy(base_config)) for _ in range(num_runs)]
            return configs, []
        
        # Generate all combinations
        param_names = list(list_params.keys())
        param_values = list(list_params.values())
        combinations = list(product(*param_values))
        
        configs = []
        for combo in combinations:
            config = copy.deepcopy(base_config)
            for param_name, param_value in zip(param_names, combo):
                ConfigLoader._set_nested_value(config, param_name, param_value)
            # Generate num_runs copies of this config (each with different random initialization)
            for _ in range(num_runs):
                configs.append(ConfigLoader(exp_path, copy.deepcopy(config)))
        
        return configs, param_names
    
    @staticmethod
    def _find_conf_static(exp_path):
        for root, _, files in os.walk(exp_path):
            for file in files:
                if "config.yaml" in file:
                    return os.path.join(root, file)
        raise ValueError("Config file was not found in provided experiment folder")
    
    # Keys whose list values are structural (not grid search dimensions)
    _grid_search_exclude = {
        "esn.W_args.W_res_args.tile.shape",
        "esn.W_args.W_back_args.range",
        "optimization.cmaes.bounds",
        "optimization.hyperneat.substrate_shape",
        "optimization.hyperneat.substrate_coords",
    }

    @staticmethod
    def _expand_folder_paths(config):
        """Expand directory paths in W_res_args.type to lists of .json tile files."""
        try:
            type_val = config["esn"]["W_args"]["W_res_args"]["type"]
        except (KeyError, TypeError):
            return

        def _expand(val):
            if isinstance(val, str) and os.path.isdir(val):
                jsons = sorted(
                    os.path.join(val, f) for f in os.listdir(val) if f.endswith(".json")
                )
                if not jsons:
                    raise ValueError(f"No .json tile files found in directory: {val}")
                return jsons
            return val

        if isinstance(type_val, list):
            expanded = []
            for v in type_val:
                result = _expand(v)
                if isinstance(result, list):
                    expanded.extend(result)
                else:
                    expanded.append(result)
            config["esn"]["W_args"]["W_res_args"]["type"] = expanded
        else:
            result = _expand(type_val)
            if isinstance(result, list):
                config["esn"]["W_args"]["W_res_args"]["type"] = result

    @staticmethod
    def _find_list_parameters(config, prefix="", result=None):
        if result is None:
            result = {}

        for key, value in config.items():
            current_path = f"{prefix}.{key}" if prefix else key

            if isinstance(value, list) and current_path not in ConfigLoader._grid_search_exclude:
                result[current_path] = value
            elif isinstance(value, dict):
                ConfigLoader._find_list_parameters(value, current_path, result)

        return result
    
    @staticmethod
    def _set_nested_value(config, path, value):
        keys = path.split(".")
        current = config
        for key in keys[:-1]:
            current = current[key]
        current[keys[-1]] = value



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
            case "Mackey-Glass":
                self.conf["dataset"] = MackeyGlass(self.conf["data"])
            case "Lorenz":
                pass
            case _:
                raise ValueError("Dataset not found")

    def _init_f(self):
        name = self.conf["esn"]["f"]["name"]
        match name:
            case "tanh" | "hysteresis":
                self.conf["esn"]["f"]["module"] = Hysteresis(**self.conf["esn"]["f"]["args"])
            case _:
                raise ValueError("F not found")

    def _init_readout(self):
        name = self.conf["esn"]["readout"]["name"]
        match name:
            case "Ridge":
                self.conf["esn"]["readout"] = Ridge(**self.conf["esn"]["readout"]["args"])
            case _:
                raise ValueError("Readout not found")

    def _init_W(self):
        w_args = self.conf["esn"]["W_args"]
        if "W_back_args" not in w_args:
            w_args["W_back_args"] = None
        self.conf["esn"]["W"] = Matrix(w_args)

    def _init_esn(self):
        self.conf["esn"]["model"] = ESN(
            W=self.conf["esn"]["W"],
            spectral_radius=self.conf["esn"]["spectral_radius"],
            f=self.conf["esn"]["f"]["module"],
            washout=self.conf["esn"]["washout"],
            readout=self.conf["esn"]["readout"],
            training_noise=self.conf["esn"].get("training_noise", 0),
            input_bias=self.conf["esn"].get("input_bias"),
        )
