# https://www.deep-ml.com/problems/59

import numpy as np

class LSTM:
	def __init__(self, input_size, hidden_size):
		self.input_size = input_size
		self.hidden_size = hidden_size

		# Initialize weights and biases
		self.Wf = np.random.randn(hidden_size, input_size + hidden_size)
		self.Wi = np.random.randn(hidden_size, input_size + hidden_size)
		self.Wc = np.random.randn(hidden_size, input_size + hidden_size)
		self.Wo = np.random.randn(hidden_size, input_size + hidden_size)

		self.bf = np.zeros((hidden_size, 1))
		self.bi = np.zeros((hidden_size, 1))
		self.bc = np.zeros((hidden_size, 1))
		self.bo = np.zeros((hidden_size, 1))

	def forward(self, x, initial_hidden_state, initial_cell_state):
		def sigmoid(val):
            return 1/(1+np.exp(-val))
        def tanh(val):
            return np.tanh(val)
        output = []
        h_t = initial_hidden_state
        c_t = initial_cell_state
        for t in range(len(x)):
            x_t = x[t].reshape(-1,1)
            concat = np.vstack((h_t,x_t))
            f_t = sigmoid(self.Wf @ concat + self.bf)
            i_t = sigmoid(self.Wi @ concat + self.bi)
            c_hat_t = tanh(self.Wc @ concat + self.bc)
            c_t = f_t * c_t + i_t * c_hat_t
            o_t = sigmoid(self.Wo @ concat + self.bo)
            h_t = o_t * tanh(c_t)
            output.append(h_t)
        return output, h_t, c_t
