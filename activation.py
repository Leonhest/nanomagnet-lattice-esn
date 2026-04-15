import torch
import torch.nn as nn


class ShiftedTanh(nn.Module):
    """Shifted tanh activation: output = tanh(beta * x - shift).

    Optionally binarizes output with sign().
    """

    def __init__(self, beta: float = 1.0, shift: float = 0.0, binary: bool = False):
        super().__init__()
        self.beta = beta
        self.shift = shift
        self.binary = binary

    def reset(self, n_nodes=None):
        pass  # No state to reset

    def forward(self, x):
        output = torch.tanh(self.beta * x - self.shift)
        if self.binary:
            output = torch.sign(output)
        return output
