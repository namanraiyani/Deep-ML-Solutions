# https://www.deep-ml.com/problems/100

def softsign(x: float) -> float:
	val = x / (1 + abs(x))
	return round(val,4)
