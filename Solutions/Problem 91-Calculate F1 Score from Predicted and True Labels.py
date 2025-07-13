# https://www.deep-ml.com/problems/91

import numpy as np
def calculate_f1_score(y_true, y_pred):
	tp = np.dot(y_true, y_pred)
	n = len(y_true)
	fp = sum([1 if y_t==0 and y_p==1 else 0 for y_t, y_p in zip(y_true, y_pred)])
	fn = sum([1 if y_t==1 and y_p==0 else 0 for y_t, y_p in zip(y_true, y_pred)])
	if tp + fp == 0 or tp + fn == 0:
		return 0.0
	precision = tp / (tp + fp)
	recall = tp / (tp + fn)
	f1 = (2 * precision * recall) / (precision + recall)
	return round(f1,3)
