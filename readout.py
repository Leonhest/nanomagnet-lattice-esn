import torch.nn as nn
from sklearn.linear_model import Ridge as SklearnRidge


class Ridge(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super(Ridge, self).__init__()
        self.alpha = alpha
        self.model = SklearnRidge(alpha=self.alpha)

    def forward(self, X, y=None):
        if y is not None:
            self.model.fit(X, y)
        return self.model.predict(X)