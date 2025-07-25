# https://www.deep-ml.com/problems/150

import numpy as np

def nag_optimizer(parameter, grad_fn, velocity, learning_rate=0.01, momentum=0.9):
    lookahead_param = parameter - momentum*velocity
    g_t = grad_fn(lookahead_param)
    velocity = momentum*velocity + learning_rate*g_t
    parameter -= velocity
    return np.round(parameter, 5), np.round(velocity, 5)
