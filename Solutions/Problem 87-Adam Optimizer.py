# https://www.deep-ml.com/problems/87

import numpy as np

def adam_optimizer(parameter, grad, m, v, t, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
	m = beta1*m + (1-beta1)*grad
    v = beta2*v + (1-beta2)*grad*grad

    m_t_hat = m / (1 - beta1)
    v_t_hat = v / (1 - beta2)

    parameter = parameter - (learning_rate)*(m_t_hat / (np.sqrt(v_t_hat) + epsilon))
	return np.round(parameter,5), np.round(m,5), np.round(v,5)
