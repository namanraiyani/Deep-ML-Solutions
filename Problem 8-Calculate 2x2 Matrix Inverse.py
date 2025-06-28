# https://www.deep-ml.com/problems/8
def inverse_2x2(matrix: list[list[float]]) -> list[list[float]]:
    det = matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]
    inverse = [[matrix[1][1]/det, -matrix[0][1]/det]], [[-matrix[1][0]/det, matrix[0][0]/det]]
	return inverse