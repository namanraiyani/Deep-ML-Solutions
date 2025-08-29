# https://www.deep-ml.com/problems/113

import numpy as np

def residual_block(x: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
	res1 = np.dot(w1, x)
	res1 = np.maximum(0,res1)
	res2 = np.dot(w2,res1)
	res2 = np.maximum(0,res2)
	out = res2+x
	out = np.maximum(0, out)
	return out
