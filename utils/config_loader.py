from re import L
import json
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

        # Check for tile replicas
        try:
            replica_files = base_config["esn"]["W_args"]["W_res_args"].get("_tile_replica_files")
        except (KeyError, TypeError):
            replica_files = None

        # Find all list parameters
        list_params = ConfigLoader._find_list_parameters(base_config)

        if not list_params:
            configs = []
            for cfg in ConfigLoader._expand_replicas(base_config, replica_files, num_runs, exp_path):
                configs.append(cfg)
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
            ConfigLoader._resolve_from_tile_params(config)
            for cfg in ConfigLoader._expand_replicas(config, replica_files, num_runs, exp_path):
                configs.append(cfg)

        return configs, param_names

    @staticmethod
    def _expand_replicas(config, replica_files, num_runs, exp_path):
        """Yield ConfigLoader instances, expanding tile replicas as an inner loop."""
        if replica_files:
            # Get the current type value (may be a directory path)
            try:
                type_val = config["esn"]["W_args"]["W_res_args"]["type"]
            except (KeyError, TypeError):
                type_val = None

            tile_files = replica_files.get(type_val, [])
            if tile_files:
                for tile_path in tile_files:
                    for _ in range(num_runs):
                        cfg = copy.deepcopy(config)
                        cfg["esn"]["W_args"]["W_res_args"]["type"] = tile_path
                        cfg["esn"]["W_args"]["W_res_args"]["_replica_group"] = type_val
                        cfg["esn"]["W_args"]["W_res_args"].pop("_tile_replica_files", None)
                        cfg["esn"]["W_args"]["W_res_args"].pop("tile_replicas", None)
                        ConfigLoader._resolve_from_tile_params(cfg)
                        yield ConfigLoader(exp_path, cfg)
                return

        # No replicas — just num_runs copies
        for _ in range(num_runs):
            cfg = copy.deepcopy(config)
            try:
                cfg["esn"]["W_args"]["W_res_args"].pop("_tile_replica_files", None)
            except (KeyError, TypeError):
                pass
            yield ConfigLoader(exp_path, cfg)
    
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

    # Keys that should be skipped entirely during list parameter discovery
    _grid_search_skip_keys = {"_tile_replica_files", "_replica_group"}

    @staticmethod
    def _expand_folder_paths(config):
        """Expand directory paths in W_res_args.type to lists of .json tile files.

        When tile_replicas is true, directories are kept as grid-search values
        and their .json contents are stored in _tile_replica_files for averaging.
        """
        try:
            W_res_args = config["esn"]["W_args"]["W_res_args"]
            type_val = W_res_args["type"]
        except (KeyError, TypeError):
            return

        tile_replicas = W_res_args.get("tile_replicas", False)

        def _list_jsons(directory):
            jsons = sorted(
                os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".json")
            )
            return jsons

        if tile_replicas:
            replica_files = {}

            if isinstance(type_val, str) and os.path.isdir(type_val):
                # Check for subdirectories with .json files
                subdirs = sorted(
                    os.path.join(type_val, d) for d in os.listdir(type_val)
                    if os.path.isdir(os.path.join(type_val, d))
                )
                subdirs_with_jsons = [d for d in subdirs if _list_jsons(d)]

                if subdirs_with_jsons:
                    # Multi-config: each subdir is a grid point
                    for subdir in subdirs_with_jsons:
                        replica_files[subdir] = _list_jsons(subdir)
                    W_res_args["type"] = subdirs_with_jsons
                else:
                    # Single-config: .json files directly in directory
                    jsons = _list_jsons(type_val)
                    if not jsons:
                        raise ValueError(f"No .json tile files found in directory: {type_val}")
                    replica_files[type_val] = jsons
                    # type stays as scalar directory path (not a grid dimension)

            elif isinstance(type_val, list):
                expanded = []
                for v in type_val:
                    if isinstance(v, str) and os.path.isdir(v):
                        jsons = _list_jsons(v)
                        if not jsons:
                            raise ValueError(f"No .json tile files found in directory: {v}")
                        replica_files[v] = jsons
                        expanded.append(v)
                    else:
                        expanded.append(v)
                W_res_args["type"] = expanded

            if replica_files:
                W_res_args["_tile_replica_files"] = replica_files
            return

        # Original behavior for non-replica mode
        def _expand(val):
            if isinstance(val, str) and os.path.isdir(val):
                jsons = _list_jsons(val)
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
            W_res_args["type"] = expanded
        else:
            result = _expand(type_val)
            if isinstance(result, list):
                W_res_args["type"] = result

    @staticmethod
    def _resolve_from_tile_params(config):
        """Resolve config values set to 'from_tile' using the tile JSON's metadata.params."""
        try:
            W_res_args = config["esn"]["W_args"]["W_res_args"]
            tile_path = W_res_args.get("type")
        except (KeyError, TypeError):
            return

        if not isinstance(tile_path, str) or not tile_path.endswith(".json"):
            return

        # Find all W_res_args values set to "from_tile"
        from_tile_keys = [k for k, v in W_res_args.items() if v == "from_tile"]
        if not from_tile_keys:
            return

        with open(tile_path, "r") as f:
            tile_data = json.load(f)
        params = tile_data.get("metadata", {}).get("params", {})

        for key in from_tile_keys:
            param_path = f"esn.W_args.W_res_args.{key}"
            if param_path in params:
                W_res_args[key] = params[param_path]

    @staticmethod
    def _find_list_parameters(config, prefix="", result=None):
        if result is None:
            result = {}

        for key, value in config.items():
            if key in ConfigLoader._grid_search_skip_keys:
                continue
            current_path = f"{prefix}.{key}" if prefix else key

            if isinstance(value, list):
                if current_path in ConfigLoader._grid_search_exclude:
                    # Nested list (e.g. shape: [[2,2], [3,3]]) → treat as sweep
                    if value and all(isinstance(v, list) for v in value):
                        result[current_path] = value
                else:
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
