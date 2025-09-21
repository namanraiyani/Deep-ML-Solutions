# https://www.deep-ml.com/problems/96

def hard_sigmoid(x: float) -> float:
	"""
	Implements the Hard Sigmoid activation function.

	Args:
		x (float): Input value

	Returns:
		float: T(he Hard Sigmoid of the input
	"""
    return round(max(float(0), min(float(1), 0.2*x + 0.5)), 3)
