# https://www.deep-ml.com/problems/66

import numpy as np
def orthogonal_projection(v, L):
    v_dot_L = np.dot(v, L)
    L_dot_L = np.dot(L, L)
    scalar = v_dot_L / L_dot_L
    projection = [scalar*Li for Li in L]
    return projection
