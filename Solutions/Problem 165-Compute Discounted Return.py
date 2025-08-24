# https://www.deep-ml.com/problems/165

import numpy as np

def discounted_return(rewards, gamma):
    ans = 0
    for i in range(len(rewards)):
        ans += rewards[i]*gamma**i
    return ans
