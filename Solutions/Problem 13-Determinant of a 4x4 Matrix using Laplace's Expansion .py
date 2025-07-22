# https://www.deep-ml.com/problems/13

def determinant_4x4(matrix: list[list[int|float]]) -> float:
	size=len(matrix)
    if size==1:
        return matrix[0][0]
    elif size==2:
        return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]
    elif size==3:
        return (
                matrix[0][0] * (matrix[1][1]*matrix[2][2] - matrix[1][2]*matrix[2][1]) -
                matrix[0][1] * (matrix[1][0]*matrix[2][2] - matrix[1][2]*matrix[2][0]) +
                matrix[0][2] * (matrix[1][0]*matrix[2][1] - matrix[1][1]*matrix[2][0])
            )
    else:
        det=0
        for col in range(size):
            minor=[
                [matrix[i][j] for j in range(size) if j!=col] for i in range(1,size)
            ]
            cofactor=(-1)**col*matrix[0][col]*determinant_4x4(minor)
            det+=cofactor
    return det
