# https://www.deep-ml.com/problems/52

import numpy as np
def recall(y_true, y_pred):
    recall_value = 0
    TP = sum([1 for y_t, y_p in zip(y_true, y_pred) if y_t==1 and y_p==1])
    FP = sum([1 for y_t, y_p in zip(y_true, y_pred) if y_t==1 and y_p==0])
    if TP+FP==0:
        return 0.0
    return round(TP/(TP+FP), 3)
