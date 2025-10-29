import torch
import numpy as np
import matplotlib.pyplot as plt



class NARMA10():

    def __init__(self, conf):
        self.conf = conf
        self.name = conf["name"]
        
        if self.conf["load"] == True:
            self._load_data()
        else:
            self._generate_data()

    def _load_data(self):
        self.u_train = torch.FloatTensor(np.load('data/datasets/NARMA10/u_train.npy'))
        self.y_train = torch.FloatTensor(np.load('data/datasets/NARMA10/y_train.npy'))
        self.u_test = torch.FloatTensor(np.load('data/datasets/NARMA10/u_test.npy'))
        self.y_test = torch.FloatTensor(np.load('data/datasets/NARMA10/y_test.npy'))

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

        u_train, y_train, u_test, y_test = self._split_data(u, y)
        self.save_data(u_train, y_train, u_test, y_test)

        self.u_train = torch.FloatTensor(u_train)
        self.y_train = torch.FloatTensor(y_train)
        self.u_test = torch.FloatTensor(u_test)
        self.y_test = torch.FloatTensor(y_test)

    def _split_data(self, u, y):
        ratio = self.conf["split_ratio"]
        u_train = u[:int(len(u) * ratio)]
        y_train = y[:int(len(y) * ratio)]
        u_test = u[int(len(u) * ratio):]
        y_test = y[int(len(y) * ratio):]
        return u_train, y_train, u_test, y_test

    def save_data(self, u_train, y_train, u_test, y_test):
        np.save('data/datasets/NARMA10/u_train.npy', u_train)
        np.save('data/datasets/NARMA10/y_train.npy', y_train)
        np.save('data/datasets/NARMA10/u_test.npy', u_test)
        np.save('data/datasets/NARMA10/y_test.npy', y_test)


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