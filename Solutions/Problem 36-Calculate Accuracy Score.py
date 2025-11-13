# https://www.deep-ml.com/problems/36

import numpy as np

def accuracy_score(y_true, y_pred):
	num = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == yp:
            num += 1
    tot = len(y_true)
    return num / tot
