# https://www.deep-ml.com/problems/74

import numpy as np

def create_row_hv(row, dim, random_seeds):
    def sign(vec):
        return np.array([1 if v>=0 else -1 for v in vec])

    def generate_hypervector(seed, dim):
        np.random.seed(seed)
        return np.random.choice([1,-1], size=dim)

    def bind(hv1, hv2):
        return hv1 * hv2

    def bundle(hvs):
        bundled = np.sum(list(hvs.values()), axis=0)
        return sign(bundled)

	hypervectors = {}

    for feature, value in row.items():
        np.random.seed(random_seeds[feature])
        feature_hv = np.random.choice([1,-1], size=dim)
        value_hv = np.random.choice([1,-1], size=dim)
        bound_hv = bind(feature_hv, value_hv)
        hypervectors[feature] = bound_hv

    composite_hv = bundle(hypervectors)
    return composite_hv
