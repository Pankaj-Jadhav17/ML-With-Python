# Write a Python Program to Multiply Two Matrices.

def multiply_matrices(matrix_a, matrix_b):
    row1 = len(matrix_a)
    col1 = len(matrix_a[0])
    row2 = len(matrix_b)
    col2 = len(matrix_b[0])

    if col1 != row2:
        raise ValueError("Incompatible matrix dimensions")

    result = [[0 for _ in range(col2)] for _ in range(row1)]

    for i in range(row1):
        for j in range(col2):
            for k in range(col1):
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]

    return result

matrix_a = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
matrix_b = [
    [9, 8, 7],
    [6, 5, 4],
    [3, 2, 1]
]
result = multiply_matrices(matrix_a, matrix_b)
if isinstance(result, str):
    print(result)
else:
    print("Result of matrix multiplication:")
    for row in result:
        print(row)
