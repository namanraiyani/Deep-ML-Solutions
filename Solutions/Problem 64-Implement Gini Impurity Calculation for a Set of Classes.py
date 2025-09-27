# https://www.deep-ml.com/problems/64

import numpy as np
from collections import Counter
def gini_impurity(y):
	"""
	Calculate Gini Impurity for a list of class labels.

	:param y: List of class labels
	:return: Gini Impurity rounded to three decimal places
	"""
    n = len(y)
    counts = Counter(y)
    gini = 1.0
    for count in counts.values():
        p = count / n
        gini -= (p**2)
	return round(gini,3)
