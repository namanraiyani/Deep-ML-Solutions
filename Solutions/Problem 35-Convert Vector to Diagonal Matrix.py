# https://www.deep-ml.com/problems/35
import numpy as np

def make_diagonal(x):
	n = len(x)
    ans = [[0]*n for _ in range(n)]
    for i in range(n):
        ans[i][i]=x[i]
    return ans
