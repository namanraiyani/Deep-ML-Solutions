# https://www.deep-ml.com/problems/159

def incremental_mean(Q_prev, k, R):
    """
    Q_prev: previous mean estimate (float)
    k: number of times the action has been selected (int)
    R: new observed reward (float)
    Returns: new mean estimate (float)
    """
    return Q_prev + ((1 / k) * (R - Q_prev))
