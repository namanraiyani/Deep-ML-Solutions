# https://www.deep-ml.com/problems/5
def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
    a,b,c,d = matrix[0][0],matrix[0][1],matrix[1][0],matrix[1][1]
    result = [[a*scalar, b*scalar], [c*scalar, d*scalar]]
	return result