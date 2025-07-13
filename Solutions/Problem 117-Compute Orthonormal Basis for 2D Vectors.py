# https://www.deep-ml.com/problems/117

import numpy as np

def orthonormal_basis(vectors: list[list[float]], tol: float = 1e-10) -> list[np.ndarray]:
    basis = []
    for v in vectors:
        v=np.array(v, dtype=float)
        for b in basis:
            v -= np.dot(v,b)*b
        norm=np.linalg.norm(v)
        if norm>tol:
            basis.append(v/norm)
    return np.array(basis)
