# https://www.deep-ml.com/problems/167

import numpy as np

def discounted_return(rewards, gamma):
    res = 0
    for i in range(len(rewards)):
        res += (gamma**i) * rewards[i]
    return res
