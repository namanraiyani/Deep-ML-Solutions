# https://www.deep-ml.com/problems/72

import numpy as np

def jaccard_index(y_true, y_pred):
	intersection = sum(1 for a,b in zip(y_true, y_pred) if a==1 and b==1)
    union = sum(1 for a,b in zip(y_true, y_pred) if a==1 or b==1)
    if union==0:
        return 0.0
    result = intersection / union
	return round(result, 3)
