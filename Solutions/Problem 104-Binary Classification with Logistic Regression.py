# https://www.deep-ml.com/problems/104

import numpy as np

def predict_logistic(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
	"""
	Implements binary classification prediction using Logistic Regression.

	Args:
		X: Input feature matrix (shape: N x D)
		weights: Model weights (shape: D)
		bias: Model bias

	Returns:
		Binary predictions (0 or 1)
	"""
    def sigmoid(arr):
        res = []
        for num in arr:
            res.append(1 / (1 + np.exp(-num)))
        return res
	z = np.dot(X, weights) + bias
    z = sigmoid(z)
    z = [1 if num>=0.5 else 0 for num in z]
    return z
