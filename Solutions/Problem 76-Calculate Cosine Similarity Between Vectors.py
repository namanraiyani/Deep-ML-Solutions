# https://www.deep-ml.com/problems/76

import numpy as np

def cosine_similarity(v1, v2):
	dot = np.dot(v1,v2)
    a=np.linalg.norm(v1)
    b=np.linalg.norm(v2)
    cos_sim = dot / (a*b)
    return round(cos_sim, 3)
