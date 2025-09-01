# https://www.deep-ml.com/problems/15

import numpy as np
def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
	m, n = X.shape
    theta = np.zeros((n,1))
    y=y.reshape(-1,1)
    for _ in range(iterations):
        predictions = np.dot(X,theta)
        error = predictions-y

        gradient = np.dot(X.T, error)/m
        theta = theta - (alpha*gradient)

	return np.round(theta.flatten(),4)
