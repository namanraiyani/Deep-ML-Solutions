# https://www.deep-ml.com/problems/98

def prelu(x: float, alpha: float = 0.25) -> float:
	return max(x, alpha * x)
