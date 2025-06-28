# https://www.deep-ml.com/problems/30
import numpy as np

def batch_iterator(X, y=None, batch_size=64):
    res = []
	if y is not None:
        n = len(y)
        for i in range(0, n, batch_size):
            res.append([X[i:i+batch_size], y[i:i+batch_size]])
    else:
        n = len(X)
        for i in range(0, n, batch_size):
            res.append(X[i:i+batch_size])
    return res