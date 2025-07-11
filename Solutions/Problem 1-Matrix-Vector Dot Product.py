# https://www.deep-ml.com/problems/1
import torch

def matrix_dot_vector(a, b) -> torch.Tensor:
    """
    Compute the product of matrix `a` and vector `b` using PyTorch.
    Inputs can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 1-D tensor of length m, or tensor(-1) if dimensions mismatch.
    """
    a_t = torch.as_tensor(a, dtype=torch.float)
    b_t = torch.as_tensor(b, dtype=torch.float)
    # Dimension mismatch check
    if a_t.size(1) != b_t.size(0):
        return torch.tensor(-1)
	ans = []
	for row in a:
		curr = 0
		k = 0
		for el in row:
			curr += el * b[k]
			k+=1
		ans.append(curr)
	return torch.tensor(ans)
