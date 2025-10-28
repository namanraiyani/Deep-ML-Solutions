v# https://www.deep-ml.com/problems/179

def dice_statistics(n: int) -> tuple[float, float]:
	"""
	Compute the expected value and variance of a fair n-sided die roll.

	Args:
		n (int): Number of sides of the die

	Returns:
		tuple: (expected_value, variance)
	"""
	expected_value = (n + 1) / 2
	variance = (n**2 - 1) / 12
	return (expected_value, variance)
