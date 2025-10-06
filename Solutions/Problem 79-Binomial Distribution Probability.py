# https://www.deep-ml.com/problems/79

import math

def binomial_probability(n, k, p):
	"""
    Calculate the probability of achieving exactly k successes in n independent Bernoulli trials,
    each with probability p of success, using the Binomial distribution formula.
    """
	binomial_coefficient = math.comb(n, k)
    probability = binomial_coefficient * (p**k) * ((1 - p) ** (n - k))
	return round(probability, 5)
