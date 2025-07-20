# https://www.deep-ml.com/problems/14

import numpy as np

def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    
    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)
    theta = XtX_inv @ X.T @ y
    theta_flat = np.ravel(theta)
    theta_rounded = [round(val,4) for val in theta_flat]
	return theta_rounded
