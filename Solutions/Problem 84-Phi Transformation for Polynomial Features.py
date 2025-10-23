# https://www.deep-ml.com/problems/84

import numpy as np

def phi_transform(data: list[float], degree: int) -> list[list[float]]:
	"""
	Perform a Phi Transformation to map input features into a higher-dimensional space by generating polynomial features.

	Args:
		data (list[float]): A list of numerical values to transform.
		degree (int): The degree of the polynomial expansion.

	"""
	ans = []
	for num in data:
		curr = []
		for i in range(degree + 1):
			curr.append(num**i)
		ans.append(curr)
	return ans
