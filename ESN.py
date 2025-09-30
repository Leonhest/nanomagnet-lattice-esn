import torch
import torch.nn as nn
from util import spectral_radius as _spectral_radius

class ESN(nn.Module):
    def __init__(self, W_in, W_res, W_out, spectral_radius, f, washout, readout):
        super(ESN, self).__init__()
        self.W_in = W_in
        self.W_res = W_res
        self.W_out = W_out
        self.spectral_radius = spectral_radius
        self.f = f
        self.washout = washout
        self.readout = readout
        self.hidden_nodes = len(self.W_res)

        if _spectral_radius(self.W_res) != 0:
                self.W_res *= self.spectral_radius / _spectral_radius(self.W_res)

    def forward(self, u, y):
        len_timeseries = len(u)
        X = torch.zeros(len_timeseries, self.hidden_nodes)
        x = torch.zeros(self.hidden_nodes)

        for t in range(len_timeseries):
            x = self.f(self.W_res @ x + self.W_in @ u[t])
            X[t] = x

        self.X = X

        X = X[self.washout:]
        y = y[self.washout:] if y is not None else None

        if y is not None:
            self.readout.fit(X, y)
        else:
            return self.readout.predict(X)

