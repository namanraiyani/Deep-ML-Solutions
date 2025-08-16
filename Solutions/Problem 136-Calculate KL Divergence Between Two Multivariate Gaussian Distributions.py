# https://www.deep-ml.com/problems/136

import numpy as np

def multivariate_kl_divergence(mu_p: np.ndarray, Cov_p: np.ndarray, mu_q: np.ndarray, Cov_q: np.ndarray) -> float:
    d = mu_p.shape[0]
    
    inv_Cov_q = np.linalg.inv(Cov_q)
    trace_term = np.trace(inv_Cov_q @ Cov_p)
    diff = mu_q - mu_p
    mahalonobis_term = diff.T @ inv_Cov_q @ diff
    log_det_ratio = np.log(np.linalg.det(Cov_q) / np.linalg.det(Cov_p))

    kl = 0.5 * (log_det_ratio - d + trace_term + mahalonobis_term)
    return kl
