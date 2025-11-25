# https://www.deep-ml.com/problems/156

import numpy as np

def SwiGLU(x: np.ndarray) -> np.ndarray:
    """
    Args:
        x: np.ndarray of shape (batch_size, 2d)
    Returns:
        np.ndarray of shape (batch_size, d)
    """
    a, b = np.split(x, 2, axis = -1)
    swish_b = b * (1 / (1 + np.exp(-b)))
    scores = a * swish_b
    return scores
