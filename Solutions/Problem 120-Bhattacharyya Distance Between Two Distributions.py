# https://www.deep-ml.com/problems/120

import numpy as np
import math

def bhattacharyya_distance(p: list[float], q: list[float]) -> float:
    if len(p) != len(q) or len(p) == 0 or len(q) == 0:
        return 0.0
    p, q = np.array(p), np.array(q)
    return -math.log(np.sum(np.sqrt(p*q)))
