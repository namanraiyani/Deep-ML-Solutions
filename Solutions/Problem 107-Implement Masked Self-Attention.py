# https://www.deep-ml.com/problems/107

import numpy as np

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray):
	"""
	Compute Query (Q), Key (K), and Value (V) matrices.
	"""
	return np.dot(X, W_q), np.dot(X, W_k), np.dot(X, W_v)

def masked_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray) -> np.ndarray:
	"""
	Compute masked self-attention.
	"""
    d_k = Q.shape[1]
	attn_scores = np.matmul(Q, K.T) / np.sqrt(d_k)
    attn_scores += mask
    attn_wts = np.exp(attn_scores - np.max(attn_scores, axis=1, keepdims=True))
    attn_wts /= np.sum(attn_wts, axis=1, keepdims=True)
    output = np.matmul(attn_wts, V)
    return output
