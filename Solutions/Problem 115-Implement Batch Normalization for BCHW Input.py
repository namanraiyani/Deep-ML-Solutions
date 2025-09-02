# https://www.deep-ml.com/problems/115

import numpy as np

def batch_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
	mu = np.mean(X, axis=(0,2,3),keepdims=True)
    var = np.var(X, axis=(0,2,3),keepdims=True)
    X_hat = (X - mu) / np.sqrt(var + epsilon)
    out = gamma * X_hat + beta
    return out
