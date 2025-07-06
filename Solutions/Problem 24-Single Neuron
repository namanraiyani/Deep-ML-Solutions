# https://www.deep-ml.com/problems/24

import math
import numpy as np
def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
    probabilities = []
    mse = 0
    for x, label in zip(features, labels):
	    y = np.dot(weights, x) + bias
        y_sigmoid = 1 / (1 + np.exp(-y))
        probabilities.append(y_sigmoid)
        mse += (y_sigmoid - label)**2
    mse /= len(labels)    
	return probabilities, mse
