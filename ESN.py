import torch
import torch.nn as nn
from utils.formula import spectral_radius as _spectral_radius
from matrix import Matrix

class ESN(nn.Module):
    def __init__(self, W: Matrix, readout, spectral_radius, f, washout):
        super(ESN, self).__init__()
        self.W = W
        self.spectral_radius = spectral_radius
        self.f = f
        self.washout = washout
        self.readout = readout
        self.hidden_nodes = len(self.W.W_res)
        if _spectral_radius(self.W.W_res) != 0:
                self.W.W_res *= self.spectral_radius / _spectral_radius(self.W.W_res)
        
    def forward(self, u, y=None):
        len_timeseries = u.size()[0]
        X = torch.zeros(len_timeseries, self.hidden_nodes)
        x = torch.zeros(self.hidden_nodes)
        

        for t in range(len_timeseries):
            x = self.f(self.W.W_in * u[t] + self.W.W_res.mv(x))
            X[t] = x

        self.X = X
        X = X[self.washout:]
        y = y[self.washout:] if y is not None else y

        y_pred = self.readout(X, y)

        return y_pred
    
