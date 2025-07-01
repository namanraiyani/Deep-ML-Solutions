# https://www.deep-ml.com/problems/85
import numpy as np

def pos_encoding(position: int, d_model: int):
	if position == 0 or d_model <= 0:
        return -1
    pos_encoding = np.zeros((position, d_model))
    for k in range(position):
        for i in np.arange(d_model // 2):
            denominator = np.power(10000, 2*i / d_model)
            pos_encoding[k][2*i] = np.sin(k / denominator)
            pos_encoding[k][2*i + 1] = np.cos(k / denominator)
	pos_encoding = pos_encoding.astype(np.float16)
	return pos_encoding
