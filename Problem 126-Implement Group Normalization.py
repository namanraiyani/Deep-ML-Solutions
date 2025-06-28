# https://www.deep-ml.com/problems/126
import numpy as np

def group_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, num_groups: int, epsilon: float = 1e-5) -> np.ndarray:
    # Your code here
    B,C,H,W = X.shape
    G = num_groups
    X = X.reshape(B, G, C//G, H, W)
    mean = np.mean(X, axis=(2,3,4), keepdims=True)
    var = np.var(X, axis=(2,3,4), keepdims=True)
    X_norm=(X-mean)/np.sqrt(var+epsilon)
    X_norm=X_norm.reshape(B,C,H,W)
    gamma=gamma.reshape(1,C,1,1)
    beta=beta.reshape(1,C,1,1)
    out = X_norm*gamma + beta
    return out