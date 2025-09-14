# https://www.deep-ml.com/problems/61

import numpy as np

def f_score(y_true, y_pred, beta):
	tp, fp, fn = 0, 0, 0
    for y_t, y_p in zip(y_true, y_pred):
        if y_t==1 and y_p==1:
            tp+=1
        elif y_t==0 and y_p==1:
            fp+=1
        elif y_t==1 and y_p==0:
            fn+=1
    precision = tp / (tp + fp) if tp+fp>0 else 0
    recall = tp / (tp + fn) if tp+fn > 0 else 0
    f1 = (1 + (beta**2)) * ((precision * recall) / ((beta*beta*precision) + recall))
    return round(f1, 3)
