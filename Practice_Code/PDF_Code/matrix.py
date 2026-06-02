def add_matrices(matrix_a, matrix_b):
    if len(matrix_a) != len(matrix_b) or len(matrix_a[0]) != len(matrix_b[0]):
        raise ValueError("Matrices must have the same dimensions for addition.")
    
    result = []
    for i in range(len(matrix_a)):
        row = []
        for j in range(len(matrix_a[0])):
            row.append(matrix_a[i][j] + matrix_b[i][j])
        result.append(row)
    
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
result = add_matrices(matrix_a, matrix_b)
if isinstance(result, str):
    print(result)
else:
    print("Result of matrix addition:")
    for row in result:
        print(row)