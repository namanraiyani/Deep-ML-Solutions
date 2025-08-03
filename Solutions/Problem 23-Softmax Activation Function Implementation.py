# https://www.deep-ml.com/problems/23

import math
import numpy as np

def softmax(scores: list[float]) -> list[float]:
	exps = [np.exp(score) for score in scores]
    denominator = sum(exps)
    probabilities = [exp/denominator for exp in exps]
	return probabilities
