# https://www.deep-ml.com/problems/7

import numpy as np

def transform_matrix(A: list[list[int|float]], T: list[list[int|float]], S: list[list[int|float]]) -> list[list[int|float]]:
    try:
        A=np.array(A,dtype=float)
        T=np.array(T,dtype=float)
        S=np.array(S,dtype=float)
        if T.shape[0]!=T.shape[1] or S.shape[0]!=S.shape[1]:
            return -1
        if A.shape[0]!=T.shape[0] or A.shape[1]!=S.shape[0]:
            return -1 
        if np.linalg.det(T)==0 or np.linalg.det(S)==0:
            return -1
        T_inv=np.linalg.inv(T)
        transformed=T_inv @ A @ S
	    return transformed.tolist()
    except np.linalg.LinAlgError:
        return -1
