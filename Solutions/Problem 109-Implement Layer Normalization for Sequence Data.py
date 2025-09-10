# https://www.deep-ml.com/problems/109

import numpy as np

def layer_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
	mean = np.mean(X, axis=-1, keepdims=True)
    var = np.var(X, axis=-1, keepdims=True)
    X_norm = (X-mean) / np.sqrt(var + epsilon)
    return gamma * X_norm + beta
