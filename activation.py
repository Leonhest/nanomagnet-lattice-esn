import torch
import torch.nn as nn

class Tanh(nn.Module):
    def __init__(self, beta: float = 1.0, shift: float = 0.0, binary: bool = False):
        super(Tanh, self).__init__()
        self.beta = beta
        self.shift = shift
        self.binary = binary

    def reset(self, n_nodes=None):
        pass

    def forward(self, x):
        out = torch.tanh(self.beta * x - self.shift)
        if self.binary:
            out = torch.sign(out)
        return out


class Hysteresis(nn.Module):
    """Preisach-style hysteresis activation with per-node state.

    Each node follows one of two shifted-tanh branches depending on whether
    the current input is ascending or descending relative to the previous
    output (Preisach relay semantics).

    When h_c=0 and m_r=1 this reduces to standard tanh(beta * x).
    """

    def __init__(self, h_c: float = 0.5, m_r: float = 1.0, beta: float = 1.0):
        super().__init__()
        self.h_c = h_c      # coercivity: half-width of hysteresis loop
        self.m_r = m_r       # remanence scaling factor
        self.beta = beta     # steepness of tanh transition
        self.prev_output = None

    def reset(self, n_nodes=None):
        if n_nodes is not None:
            self.prev_output = torch.zeros(n_nodes)
        elif self.prev_output is not None:
            self.prev_output = torch.zeros_like(self.prev_output)

    def forward(self, x):
        if self.prev_output is None:
            self.prev_output = torch.zeros_like(x)

        ascending = x >= self.prev_output
        y_upper = self.m_r * torch.tanh(self.beta * (x - self.h_c))
        y_lower = self.m_r * torch.tanh(self.beta * (x + self.h_c))

        output = torch.where(ascending, y_upper, y_lower)
        self.prev_output = output.detach()
        return output