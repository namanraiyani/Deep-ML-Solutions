# https://www.deep-ml.com/problems/111

import numpy as np

def compute_pmi(joint_counts, total_counts_x, total_counts_y, total_samples):
	res = np.log2((joint_counts/total_samples)/((total_counts_x/total_samples)*(total_counts_y/total_samples)))
    return round(res, 3)
