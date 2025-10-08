import torch
import torch.nn as nn

class Tanh(nn.Module):
    def __init__(self, beta: float = 1.0):
        super(Tanh, self).__init__()
        self.beta = beta

    def forward(self, x):
        return torch.tanh(self.beta * x)