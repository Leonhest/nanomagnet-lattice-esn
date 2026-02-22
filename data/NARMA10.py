import torch
import numpy as np
import matplotlib.pyplot as plt



class NARMA10():

    def __init__(self, conf):
        self.conf = conf
        self.name = conf["name"]
        self._data_dir = f'data/datasets/NARMA{conf["system_order"]}'

        if self.conf["load"] == True:
            self._load_data()
        else:
            max_attempts = 10
            for attempt in range(max_attempts):
                try:
                    self._generate_data()
                    break
                except Exception:
                    if attempt == max_attempts - 1:
                        raise

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
        n = self.conf["system_order"]
        sample_len = self.conf["sample_len"]

        if n == 10 or n == 20:
            alpha = 0.3
            beta = 0.05
            gamma = 1.5
            delta = 0.1
        elif n == 30:
            # Unchanged for now.
            alpha = 0.3
            beta = 0.05
            gamma = 1.5
            delta = 0.1
        else:
            raise ValueError('Invalid system order for NARMA time series')

        u = torch.rand(sample_len) * 0.5
        y = torch.zeros(sample_len)

        for t in range(n, sample_len):
            y[t] = alpha*y[t-1] + \
                beta*y[t-1]*torch.sum(y[t-n:t]) + \
                gamma*u[t-1]*u[t-n] + \
                delta
            if n != 10:
                y[t] = np.tanh(y[t])

        if not np.isfinite(y).all():
            class DivergentTimeseriesError(Exception):
                pass
            raise DivergentTimeseriesError('Divergent NARMA time series, try again')

        splits = self._split_data(u, y)
        self.save_data(**splits)

        self.u_train = torch.FloatTensor(splits["u_train"])
        self.y_train = torch.FloatTensor(splits["y_train"])
        self.u_test = torch.FloatTensor(splits["u_test"])
        self.y_test = torch.FloatTensor(splits["y_test"])

        if "u_val" in splits:
            self.u_val = torch.FloatTensor(splits["u_val"])
            self.y_val = torch.FloatTensor(splits["y_val"])

    def _split_data(self, u, y):
        ratio = self.conf["split_ratio"]
        split_idx = int(len(u) * ratio)
        u_trainval = u[:split_idx]
        y_trainval = y[:split_idx]
        u_test = u[split_idx:]
        y_test = y[split_idx:]

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


if __name__ == "__main__":
    # load data from file
    u_train = np.load('datasets/NARMA10/u_train.npy')
    y_train = np.load('datasets/NARMA10/y_train.npy')
    u_test = np.load('datasets/NARMA10/u_test.npy')
    y_test = np.load('datasets/NARMA10/y_test.npy')

    # plot distribution of u_train and y_train
    plt.hist(u_train, bins=100)
    plt.show()
    plt.hist(y_train, bins=100)
    plt.show()
    plt.hist(u_test, bins=100)
    plt.show()
    plt.hist(y_test, bins=100)
    plt.show()