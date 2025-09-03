# https://www.deep-ml.com/problems/178

import numpy as np

def ffn(x: list[float], W1: list[list[float]], b1: list[float], W2: list[list[float]], b2: list[float], dropout_p: float=0.1, seed: int=42) -> list[float]:
	"""
	Implement a position-wise feed-forward block with residual and dropout.

	Args:
		x: input vector
		W1, b1: first linear layer parameters
		W2, b2: second linear layer parameters
		dropout_p: dropout probability
		seed: random seed for reproducibility

	Returns:
		Output vector after FFN block (rounded to 4 decimals)
	"""
    np.random.seed(seed)
	x = np.array(x)
    w1 = np.array(W1)
    b1 = np.array(b1)
    w2 = np.array(W2)
    b2 = np.array(b2)
    hidden = np.dot(w1, x) + b1
    hidden = np.maximum(0, hidden)
    output = np.dot(w2, hidden) + b2
    output += x
    dropout_mask=np.random.binomial(1,1-dropout_p, size=output.shape)
    output *= dropout_mask / (1-dropout_p)
    return list(np.round(output, 4))
