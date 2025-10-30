# https://www.deep-ml.com/problems/154

class ExponentialLRScheduler:
    def __init__(self, initial_lr, gamma):
        self.initial_lr = initial_lr
        self.gamma = gamma

    def get_lr(self, epoch):
        # Calculate and return the learning rate for the given epoch
        return self.initial_lr*(self.gamma**epoch)
