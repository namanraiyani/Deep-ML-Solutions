# https://www.deep-ml.com/problems/53

import numpy as np

def compute_qkv(X, W_q, W_k, W_v):
	Q, K, V = X @ W_q, X @ W_k, X @ W_v
	return Q, K, V

def self_attention(Q, K, V):
	def softmax(x):
		exps = np.exp(x - np.max(x, axis=-1, keepdims = True))
		return exps / np.sum(exps, axis=-1, keepdims=True)
	d_k = Q.shape[-1]
	scores = Q @ K.T / np.sqrt(d_k)
	weights = softmax(scores)
	output = weights @ V
	return output
