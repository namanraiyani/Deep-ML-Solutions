# https://www.deep-ml.com/problems/29
import numpy as np

def shuffle_data(X, y, seed=None):
    if seed is not None:
	    np.random.seed(seed)
	permutation = np.random.permutation(len(X))
    return X[permutation], y[permutation]
