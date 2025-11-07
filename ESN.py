import torch
import torch.nn as nn
from utils.formula import spectral_radius as _spectral_radius
from matrix import Matrix
import numpy as np
import logging
logger = logging.getLogger(__name__)
from readout import Ridge 
class ESN(nn.Module):
    def __init__(self, W: Matrix, readout, spectral_radius, f, washout):
        super(ESN, self).__init__()
        self.W = W
        self.spectral_radius = spectral_radius
        self.f = f
        self.washout = washout
        self.readout = readout
        self.hidden_nodes = len(self.W.W_res)
        spec_rad = _spectral_radius(self.W.W_res)
        if spec_rad != 0:
            self.W.W_res *= self.spectral_radius / spec_rad
    def forward(self, u, y=None, kq=False):
        len_timeseries = u.size()[0]
        X = torch.zeros(len_timeseries, self.hidden_nodes)
        x = torch.zeros(self.hidden_nodes)
        

        for t in range(len_timeseries):
            x = self.f(self.W.W_in * u[t] + self.W.W_res.mv(x))
            X[t] = x

        self.X = X
        X = X[self.washout:]
        y = y[self.washout:] if y is not None else y

        if not kq:
            y_pred = self.readout(X, y)
            return y_pred
        else:
            # When kq=True, we just need to store X, don't compute readout
            return None
    
    def memory_capacity(self, washout, u_train, u_test, plot=False):

        output_nodes = int(1.4*self.hidden_nodes)
        washout_len = washout.shape[0]
        train_len = u_train.shape[0]

        self(torch.cat((washout, u_train, u_test), 0), kq=True)
        self.X_train = self.X[washout_len:washout_len+train_len]

        self.w_outs = [0]*output_nodes


        for k in range(1, output_nodes+1):
            self.w_outs[k-1] = Ridge()
            self.w_outs[k-1](self.X_train[k:, :], u_train[:-k])
            

        self.X_test = self.X[washout_len+train_len:]

        ys = torch.FloatTensor([rr(self.X_test) for rr in self.w_outs])
      

        if plot:
            import matplotlib.pyplot as plt
            plt.plot(u_test)
            for k in range(output_nodes):
                plt.plot(ys[k][k+1:])
            plt.show()

        mc = 0
        self.mcs = []
        for k in range(output_nodes):
            u_tk = u_test[:-(k+1)]
            numerator = (np.cov(u_tk, ys[k][k+1:], bias=True)[0][1])**2
            denominator = torch.var(u_tk)*torch.var(ys[k][k+1:])
            _mc = numerator/denominator
            mc += _mc
            self.mcs.append(_mc)
        return float(mc)
