# https://www.deep-ml.com/problems/151

import numpy as np

class DropoutLayer:
    def __init__(self, p: float):
        """Initialize the dropout layer."""
        self.p = p
        self.mask = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass of the dropout layer."""
        if not training:
            return x
        self.mask = np.random.binomial(1, 1 - self.p, x.shape)
        return x * self.mask / (1 - self.p)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass of the dropout layer."""
        return grad * self.mask / (1 - self.p)
