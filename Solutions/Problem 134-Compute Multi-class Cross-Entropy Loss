# https://www.deep-ml.com/problems/134

import numpy as np

def compute_cross_entropy_loss(predicted_probs: np.ndarray, true_labels: np.ndarray, epsilon = 1e-15) -> float:
    multi_cross_entropy_loss = 0
    for probs, labels in zip(predicted_probs, true_labels):
        multi_cross_entropy_loss -= np.log(np.dot(probs, labels))
    return multi_cross_entropy_loss / len(predicted_probs)
