# https://www.deep-ml.com/problems/43

import numpy as np

def ridge_loss(X: np.ndarray, w: np.ndarray, y_true: np.ndarray, alpha: float) -> float:
    y_pred = X @ w
    mse_loss = np.mean((y_true - y_pred)**2)
    reg_term = alpha * np.sum(w**2)
    return mse_loss + reg_term
