# https://www.deep-ml.com/problems/6
import numpy
def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
	trace = matrix[0][0] + matrix[1][1]
    det = matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]
    lambda_1 = (trace + numpy.sqrt(trace**2 - 4*det)) / (2)
    lambda_2 = (trace - numpy.sqrt(trace**2 - 4*det)) / (2)
    return [lambda_1, lambda_2]