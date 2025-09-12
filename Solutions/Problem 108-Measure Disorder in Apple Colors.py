# https://www.deep-ml.com/problems/108

import numpy as np

def disorder(apples: list) -> float:
	apples = np.array(apples)
    var = np.var(apples)
    return var
