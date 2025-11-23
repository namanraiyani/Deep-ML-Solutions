# https://www.deep-ml.com/problems/11

import numpy as np
def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
    diag_el = np.diag(A)
    off_diag = A - np.diag(diag_el)
    x = np.zeros(len(b))
    x_temp = np.zeros(len(b))
    for _ in range(n):
        for i in range(len(A)):
            x_temp[i] = (1 / diag_el[i]) * (b[i] - np.dot(off_diag[i], x))
        x = x_temp.copy()
    x = np.round(x, 4).tolist()
	return x
