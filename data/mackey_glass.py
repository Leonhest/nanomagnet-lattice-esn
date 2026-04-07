import torch
import numpy as np


class MackeyGlass():

    def __init__(self, conf):
        self.conf = conf
        self.name = conf["name"]
        tau = conf.get("tau", 17)
        self._data_dir = f'data/datasets/MackeyGlass_tau{tau}'

        if self.conf["load"] == True:
            self._load_data()
        else:
            self._generate_data()

    def _load_data(self):
        d = self._data_dir
        self.u_train = torch.FloatTensor(np.load(f'{d}/u_train.npy'))
        self.y_train = torch.FloatTensor(np.load(f'{d}/y_train.npy'))
        self.u_test = torch.FloatTensor(np.load(f'{d}/u_test.npy'))
        self.y_test = torch.FloatTensor(np.load(f'{d}/y_test.npy'))

        if "val_ratio" in self.conf and self.conf["val_ratio"]:
            self.u_val = torch.FloatTensor(np.load(f'{d}/u_val.npy'))
            self.y_val = torch.FloatTensor(np.load(f'{d}/y_val.npy'))

    def _generate_data(self):
        tau = self.conf.get("tau", 17)
        if "train_len" in self.conf and "test_len" in self.conf:
            sample_len = self.conf["train_len"] + self.conf["test_len"]
        else:
            sample_len = self.conf["sample_len"]
        h = 1 if self.conf.get("closed_loop", False) else self.conf.get("prediction_horizon", 1)
        warmup = 1000

        # Standard Mackey-Glass parameters (Jaeger 2001, Holzmann & Hauser 2010)
        alpha = 0.2
        gamma = 0.1
        n = 10
        delta = 0.1  # Euler stepsize
        tau_steps = int(tau / delta)  # delay in fine steps

        # Total coarse steps needed: warmup + sample_len + h
        total_coarse = warmup + sample_len + h
        total_fine = total_coarse * int(1 / delta)  # 10x finer

        # Initialize history with x = 1.2 (Jaeger's initial condition)
        x = np.zeros(total_fine + tau_steps)
        x[:tau_steps + 1] = 1.2

        # Euler integration at fine timescale (delta = 0.1)
        for t in range(tau_steps, total_fine + tau_steps - 1):
            x_t = x[t]
            x_d = x[t - tau_steps]
            x[t + 1] = x_t + delta * (alpha * x_d / (1.0 + x_d**n) - gamma * x_t)

        # Subsample by 10 to get unit time steps, discard history and warmup
        subsample = int(1 / delta)
        series = x[tau_steps::subsample]
        series = series[warmup:warmup + sample_len + h]

        # Squash into [-1, 1] range (Jaeger: y -> tanh(y - 1))
        series = np.tanh(series - 1)

        # u[t] = x[t], y[t] = x[t + h]
        u = torch.from_numpy(series[:sample_len]).float()
        y = torch.from_numpy(series[h:sample_len + h]).float()

        splits = self._split_data(u, y)
        if self.conf.get("save", False):
            self.save_data(**splits)

        self.u_train = torch.FloatTensor(splits["u_train"])
        self.y_train = torch.FloatTensor(splits["y_train"])
        self.u_test = torch.FloatTensor(splits["u_test"])
        self.y_test = torch.FloatTensor(splits["y_test"])

        if "u_val" in splits:
            self.u_val = torch.FloatTensor(splits["u_val"])
            self.y_val = torch.FloatTensor(splits["y_val"])

    def _split_data(self, u, y):
        # Support explicit train_len/test_len or fall back to split_ratio
        train_len = self.conf.get("train_len")
        test_len = self.conf.get("test_len")

        if train_len is not None and test_len is not None:
            split_idx = train_len
        else:
            ratio = self.conf["split_ratio"]
            split_idx = int(len(u) * ratio)

        u_trainval = u[:split_idx]
        y_trainval = y[:split_idx]
        u_test = u[split_idx:split_idx + test_len] if test_len is not None else u[split_idx:]
        y_test = y[split_idx:split_idx + test_len] if test_len is not None else y[split_idx:]

        val_ratio = self.conf.get("val_ratio")
        if val_ratio:
            val_idx = int(len(u_trainval) * (1 - val_ratio))
            u_train = u_trainval[:val_idx]
            y_train = y_trainval[:val_idx]
            u_val = u_trainval[val_idx:]
            y_val = y_trainval[val_idx:]
            return {"u_train": u_train, "y_train": y_train,
                    "u_val": u_val, "y_val": y_val,
                    "u_test": u_test, "y_test": y_test}

        return {"u_train": u_trainval, "y_train": y_trainval,
                "u_test": u_test, "y_test": y_test}

    def save_data(self, u_train, y_train, u_test, y_test, u_val=None, y_val=None):
        import os
        os.makedirs(self._data_dir, exist_ok=True)
        d = self._data_dir
        np.save(f'{d}/u_train.npy', u_train)
        np.save(f'{d}/y_train.npy', y_train)
        np.save(f'{d}/u_test.npy', u_test)
        np.save(f'{d}/y_test.npy', y_test)
        if u_val is not None:
            np.save(f'{d}/u_val.npy', u_val)
            np.save(f'{d}/y_val.npy', y_val)
