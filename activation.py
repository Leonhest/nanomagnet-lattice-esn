import torch
import torch.nn as nn


class Hysteresis(nn.Module):
    """Preisach-style hysteresis activation with per-node state and minor loops.

    Each node follows one of two shifted-tanh branches depending on whether
    the current input is ascending or descending. On direction reversal, the
    output transitions smoothly from the current position toward the target
    major branch via an exponentially decaying offset, producing minor loops.

    When h_c=0, m_r=1, shift=0, decay_rate=0 this reduces to standard tanh(beta * x).
    """

    def __init__(self, h_c: float = 0.0, m_r: float = 1.0, beta: float = 1.0,
                 shift: float = 0.0, binary: bool = False, decay_rate: float = 2.0):
        super().__init__()
        self.h_c = h_c            # coercivity: half-width of hysteresis loop
        self.m_r = m_r             # remanence scaling factor
        self.beta = beta           # steepness of tanh transition
        self.shift = shift         # horizontal shift (bias field)
        self.binary = binary       # binarize output with sign()
        self.decay_rate = decay_rate  # minor loop merge speed (0 = hard switch)
        self.prev_output = None
        self.prev_x = None
        self.was_ascending = None
        self.reversal_x = None
        self.branch_offset = None

    def reset(self, n_nodes=None):
        if n_nodes is not None:
            self.prev_output = torch.zeros(n_nodes)
            self.prev_x = torch.zeros(n_nodes)
            self.was_ascending = torch.ones(n_nodes, dtype=torch.bool)
            self.reversal_x = torch.zeros(n_nodes)
            self.branch_offset = torch.zeros(n_nodes)
        elif self.prev_output is not None:
            self.prev_output = torch.zeros_like(self.prev_output)
            self.prev_x = torch.zeros_like(self.prev_x)
            self.was_ascending = torch.ones_like(self.was_ascending)
            self.reversal_x = torch.zeros_like(self.reversal_x)
            self.branch_offset = torch.zeros_like(self.branch_offset)

    def _upper(self, x):
        return self.m_r * torch.tanh(self.beta * (x - self.h_c) - self.shift)

    def _lower(self, x):
        return self.m_r * torch.tanh(self.beta * (x + self.h_c) - self.shift)

    def forward(self, x):
        if self.prev_output is None:
            self.prev_output = torch.zeros_like(x)
            self.prev_x = torch.zeros_like(x)
            self.was_ascending = torch.ones(x.shape[0], dtype=torch.bool)
            self.reversal_x = torch.zeros_like(x)
            self.branch_offset = torch.zeros_like(x)

        ascending = x >= self.prev_x
        y_target = torch.where(ascending, self._upper(x), self._lower(x))

        if self.h_c == 0 or self.decay_rate == 0:
            # No hysteresis or hard switch — skip offset logic
            output = y_target
        else:
            # Detect direction reversals
            reversed_dir = ascending != self.was_ascending

            # At reversal points, compute offset for continuity
            y_target_at_rev = torch.where(
                ascending, self._upper(self.prev_x), self._lower(self.prev_x)
            )
            new_offset = self.prev_output - y_target_at_rev
            self.branch_offset = torch.where(reversed_dir, new_offset, self.branch_offset)
            self.reversal_x = torch.where(reversed_dir, self.prev_x, self.reversal_x)

            # Decay offset with distance from reversal point
            dist = torch.abs(x - self.reversal_x)
            decay = torch.exp(-self.decay_rate * dist)
            output = (y_target + self.branch_offset * decay).clamp(-self.m_r, self.m_r)

        if self.binary:
            output = torch.sign(output)

        self.prev_output = output.detach()
        self.prev_x = x.detach()
        self.was_ascending = ascending
        return output
