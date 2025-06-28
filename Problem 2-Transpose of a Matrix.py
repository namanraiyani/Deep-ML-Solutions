# https://www.deep-ml.com/problems/2
import torch

def transpose_matrix(a) -> torch.Tensor:
    """
    Transpose a 2D matrix `a` using PyTorch.
    Inputs can be Python lists, NumPy arrays, or torch Tensors.
    Returns a transposed tensor.
    """
    a_t = torch.as_tensor(a)
    # Your implementation here
    return torch.transpose(a_t, 0, 1
)