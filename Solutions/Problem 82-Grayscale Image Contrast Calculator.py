# https://www.deep-ml.com/problems/82

import numpy as np

def calculate_contrast(img) -> int:
	return np.max(img) - np.min(img)
