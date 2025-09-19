# https://www.deep-ml.com/problems/39

import numpy as np

def log_softmax(scores: list) -> np.ndarray:
    scores = np.array(scores)
    max_score = np.max(scores)
    exp_scores = np.exp(scores - max_score)
    sum_exp_scores = np.sum(exp_scores)
    log_probs = scores - max_score - np.log(sum_exp_scores)
    return np.round(log_probs, 4)
