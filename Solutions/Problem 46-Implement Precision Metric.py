# https://www.deep-ml.com/problems/46

import numpy as np

def precision(y_true, y_pred):
    tp, fp = 0, 0
    for yt, yp in zip(y_true, y_pred):
        if yt==1 and yp==1:
            tp+=1
        if yp==1 and yt==0:
            fp+=1
    precision = tp/(tp+fp)
    return round(precision,3)
