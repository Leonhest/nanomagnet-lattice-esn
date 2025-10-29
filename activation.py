import torch
import torch.nn as nn

class Tanh(nn.Module):
    def __init__(self, beta: float = 1.0, shift: float = 0.0):
        super(Tanh, self).__init__()
        self.beta = beta
        self.shift = shift

    def forward(self, x):
        return torch.tanh(self.beta * x - self.shift)