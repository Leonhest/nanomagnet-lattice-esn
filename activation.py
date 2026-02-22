import torch
import torch.nn as nn

class Tanh(nn.Module):
    def __init__(self, beta: float = 1.0, shift: float = 0.0, binary: bool = False):
        super(Tanh, self).__init__()
        self.beta = beta
        self.shift = shift
        self.binary = binary

    def forward(self, x):
        out = torch.tanh(self.beta * x - self.shift)
        if self.binary:
            out = torch.sign(out)
        return out