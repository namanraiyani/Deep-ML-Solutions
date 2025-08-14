# https://www.deep-ml.com/problems/94

import numpy as np
import math

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))  
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def compute_qkv(X, W_q, W_k, W_v):
	Q = np.dot(X, W_q)
    K = np.dot(X, W_k)
    V = np.dot(X, W_v)
    return Q, K, V

def self_attention(Q, K, V):
	scores = np.dot(Q, K.T)
    d_k = Q.shape[-1]
    scaled_scores = (scores)/math.sqrt(d_k)
    attention_weights = softmax(scaled_scores, axis=-1)
    output = np.dot(attention_weights, V)
    return output

def multi_head_attention(Q, K, V, n_heads):
    d_model = Q.shape[-1]
    d_k = d_model // n_heads
    Q_split = np.split(Q, n_heads, axis=-1)
    K_split = np.split(K, n_heads, axis=-1)
	V_split = np.split(V, n_heads, axis=-1)
    outputs = []
    for i in range(n_heads):
        Q_i = Q_split[i]
        K_i = K_split[i]
        V_i = V_split[i]
        output_i = self_attention(Q_i, K_i, V_i)
        outputs.append(output_i)
    multi_head_output = np.concatenate(outputs, axis=-1)
    return multi_head_output
