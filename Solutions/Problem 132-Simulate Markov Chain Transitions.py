# https://www.deep-ml.com/problems/132

import numpy as np
def simulate_markov_chain(transition_matrix, initial_state, num_steps):
    states = [initial_state]
    curr = initial_state
    for i in range(num_steps):
        curr = np.random.choice(
            a = len(transition_matrix),
            p = transition_matrix[curr]
        )
        states.append(curr)
    return np.array(states)
