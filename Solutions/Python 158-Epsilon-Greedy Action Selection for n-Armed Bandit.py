# https://www.deep-ml.com/problems/158

import numpy as np

def epsilon_greedy(Q, epsilon=0.1):
    """
    Selects an action using epsilon-greedy policy.
    Q: np.ndarray of shape (n,) -- estimated action values
    epsilon: float in [0, 1]
    Returns: int, selected action index
    """
    if np.random.rand() < epsilon:
        action = np.random.choice(len(Q))
    else:
        action = np.argmax(Q)
    return action
