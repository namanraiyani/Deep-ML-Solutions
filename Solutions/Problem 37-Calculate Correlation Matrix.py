# https://www.deep-ml.com/problems/37

import numpy as np

def calculate_correlation_matrix(X, Y=None):
    if Y is None:
        return np.corrcoef(X.T)
	return np.corrcoef(X.T, Y.T)[:X.shape[1], X.shape[1]:]
