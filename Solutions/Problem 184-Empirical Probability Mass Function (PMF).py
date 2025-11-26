# https://www.deep-ml.com/problems/184

from collections import Counter

def empirical_pmf(samples):
    """
    Given an iterable of integer samples, return a list of (value, probability)
    pairs sorted by value ascending.
    """
    counts = Counter(samples)
    n = len(samples)
    ans = []
    for val, count in counts.items():
        ans.append((val, count / n))
    return sorted(ans)
