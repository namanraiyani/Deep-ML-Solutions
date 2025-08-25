# https://www.deep-ml.com/problems/119

import numpy as np

def cramers_rule(A, b):
    A = np.array(A)
    b = np.array(b)
    det_A = np.linalg.det(A)
    if det_A == 0:
        return -1
    x = []
    for i in range(len(b)):
        A1 = A.copy()
        A1[:, i] = b
        det_A1 = np.linalg.det(A1)
        x.append(det_A1 / det_A)
    return np.array(x)
