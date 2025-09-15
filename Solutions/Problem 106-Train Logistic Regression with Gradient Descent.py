# https://www.deep-ml.com/problems/106

import numpy as np

def train_logreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
	def sigmoid(z):
        return 1 / (1 + np.exp(-z))
    def binary_cross_entropy(y_true, y_pred):
        return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    n_samples, n_features = X.shape
    X = np.hstack((np.ones((n_samples, 1)), X))
	W = np.zeros(n_features + 1)
    loss_history = []
    for i in range(iterations):
        linear_model = np.dot(X, W)
        y_pred = sigmoid(linear_model)
        
		loss = binary_cross_entropy(y, y_pred)
        loss_history.append(round(loss, 4))

		gradient = np.dot(X.T, (y_pred - y))        
		W -= learning_rate * gradient

	coeff = W.round(4).tolist()
	return coeff, loss_history
