# https://www.deep-ml.com/problems/141

import numpy as np

def convert_range(values: np.ndarray, c: float, d: float) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    a, b = np.min(x), np.max(x)
    if a==b:
        return np.full_like(x, (c + d)/2, dtype = float)
    return (((x-a) / (b-a)) * (d-c)) + c
