# https://www.deep-ml.com/problems/27
import numpy as np

def transform_basis(B: list[list[int]], C: list[list[int]]) -> list[list[float]]:
    B = np.array(B)
    C = np.array(C)

    C_inv = np.linalg.inv(C)

    P = np.dot(C_inv, B)
	return P
