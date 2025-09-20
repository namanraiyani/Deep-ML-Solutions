import numpy as np

def grpo_objective(rhos, A, pi_theta_old, pi_theta_ref, epsilon=0.2, beta=0.01) -> float:
	"""
	Compute the GRPO objective function.

	Args:
		rhos: List of likelihood ratios (p_i) = pi_theta(o_i | q) / pi_theta_old(o_i | q).
		A: List of advantage estimates (A_i).
		pi_theta_old: List representing the old policy probabilities pi_theta_old(o_i | q).
		pi_theta_ref: List representing the reference policy probabilities pi_ref(o_i | q).
		epsilon: Clipping parameter (eps).
		beta: KL divergence penalty coefficient (beta).

	Returns:
		The computed GRPO objective value.
	"""
    rhos = np.array(rhos)
    A = np.array(A)
    pi_theta_old = np.array(pi_theta_old)
    pi_theta_ref = np.array(pi_theta_ref)
	clipped_rho = np.clip(rhos, 1 - epsilon, 1 + epsilon)
    
    unclipped = rhos * A
    clipped = clipped_rho * A
    min_terms = np.minimum(unclipped, clipped)
    average_min = np.mean(min_terms)
    pi_theta = rhos * pi_theta_old
    pi_theta /= np.sum(pi_theta)
    pi_theta_ref /= np.sum(pi_theta_ref)
    
    kl_divergence = np.sum(pi_theta * np.log(pi_theta / pi_theta_ref + 1e-10))
    objective = average_min - beta * kl_divergence
    return objective
