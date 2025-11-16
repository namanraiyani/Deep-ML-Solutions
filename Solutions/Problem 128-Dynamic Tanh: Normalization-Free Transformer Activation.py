# https://www.deep-ml.com/problems/128

import numpy as np

def dynamic_tanh(x: np.ndarray, alpha: float, gamma: float, beta: float) -> list[float]:
    transformed = alpha * x + beta
    y = np.tanh(transformed) * gamma
    return y.tolist()
