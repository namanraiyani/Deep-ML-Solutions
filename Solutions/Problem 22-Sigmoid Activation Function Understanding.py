# https://www.deep-ml.com/problems/22

import math

def sigmoid(z: float) -> float:
	result = 1 / (1 + math.exp(-z))
	return round(result, 4)
