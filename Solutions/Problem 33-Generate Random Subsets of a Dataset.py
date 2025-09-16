# https://www.deep-ml.com/problems/33

import numpy as np

def get_random_subsets(X, y, n_subsets, replacements=True, seed=42):
	subsets = []
    n_samples = X.shape[0]
    np.random.seed(seed)
    subset_size = n_samples if replacements else 2
    for _ in range(n_subsets):
        indices = np.random.choice(n_samples, size = subset_size, replace = replacements)
        X_subset = X[indices].tolist()
        y_subset = y[indices].tolist()
        subsets.append((X_subset, y_subset))
    return subsets
