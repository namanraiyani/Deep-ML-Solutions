# https://www.deep-ml.com/problems/93
import numpy as np

def mae(y_true, y_pred):
	return round(np.mean(np.abs(y_true - y_pred)),3)