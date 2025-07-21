# https://www.deep-ml.com/problems/123

def compute_efficiency(n_experts, k_active, d_in, d_out):
    savings = (1 - (k_active / n_experts)) * 100
    return savings
