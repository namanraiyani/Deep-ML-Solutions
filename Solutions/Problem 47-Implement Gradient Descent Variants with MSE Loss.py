# https://www.deep-ml.com/problems/47

import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_iterations, batch_size=1, method='batch'):
	m = len(y)
	for _ in range(n_iterations):
		if method == 'batch':
			pred = X.dot(weights)
			err = pred - y
			grads = (2/m) * X.T.dot(err)
			weights -= learning_rate * grads
		elif method == 'stochastic':
			for j in range(m):
				x_j = X[j : j+1]
				y_j = y[j : j+1]
				pred = x_j.dot(weights)
				err = pred - y_j
				grads = 2*x_j.T.dot(err)
				weights -= learning_rate * grads
		else: # mini-batch
			for j in range(0, m, batch_size):
				x_j_batch = X[j : j + batch_size]
				y_j_batch = y[j : j + batch_size]
				preds = x_j_batch.dot(weights)
				err = preds - y_j_batch
				grads = (2/batch_size) * x_j_batch.T.dot(err)
				weights -= learning_rate * grads
	return weights
