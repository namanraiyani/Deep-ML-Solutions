# https://www.deep-ml.com/problems/163

import numpy as np

class GradientBandit:
    def __init__(self, num_actions, alpha=0.1):
        """
        num_actions (int): Number of possible actions
        alpha (float): Step size for preference updates
        """
        self.num_actions = num_actions
        self.alpha = alpha
        self.preferences = np.zeros(num_actions)
        self.avg_reward = 0.0
        self.time = 0

    def softmax(self):
        e_x = np.exp(self.preferences - np.max(self.preferences))
        return e_x / np.sum(e_x)

    def select_action(self):
        self.probs = self.softmax()
        return np.random.choice(self.num_actions, p = self.probs)

    def update(self, action, reward):
        self.time += 1
        self.avg_reward  += (reward - self.avg_reward) / self.time
        probs = self.softmax()

        baseline = self.avg_reward
        for a in range(self.num_actions):
            indicator = 1.0 if a==action else 0.0
            self.preferences[a] += self.alpha * (reward - baseline) * (indicator - probs[a])
