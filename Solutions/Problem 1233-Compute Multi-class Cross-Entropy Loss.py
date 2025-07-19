# https://www.deep-ml.com/problems/134

import numpy as np

def compute_cross_entropy_loss(predicted_probs: np.ndarray, true_labels: np.ndarray, epsilon = 1e-15) -> float:
    loss = 0
    for p, t in zip(predicted_probs, true_labels):
        for prob, label in zip(p,t):
            if label!=0:
                loss -= np.log(max(epsilon, min(prob, 1-epsilon)))
        
    loss /= len(true_labels)
    return loss
