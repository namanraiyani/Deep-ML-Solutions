# https://www.deep-ml.com/problems/50

import numpy as np

def l1_regularization_gradient_descent(X: np.array, y: np.array, alpha: float = 0.1,
                                       learning_rate: float = 0.01, max_iter: int = 1000,
                                       tol: float = 1e-6) -> tuple:
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0
    for _ in range(max_iter):
        y_pred = X.dot(weights) + bias
        error = y_pred - y
        dw = (1/n_samples) * X.T.dot(error) + alpha * np.sign(weights)
        db = (1/n_samples) * np.sum(error)
        weights_old = weights.copy()
        weights -= learning_rate * dw
        bias -= learning_rate * db
        if np.linalg.norm(weights - weights_old, ord=1) < tol:
            break

    return weights, bias
