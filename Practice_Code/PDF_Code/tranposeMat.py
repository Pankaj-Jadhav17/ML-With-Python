# Write a Python Program to Transpose a Matrix.
def transpose_matrix(matrix):
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

# Example usage:
if __name__ == "__main__":
    # Define a matrix
    matrix = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]

    # Transpose the matrix
    transposed = transpose_matrix(matrix)

    # Print the original and transposed matrices
    print("Original Matrix:")
    for row in matrix:
        print(row)

    print("\nTransposed Matrix:")
    for row in transposed:
        print(row)
        
        
#         Function to transpose a matrix
# def transpose_matrix(matrix):
# rows, cols = len(matrix), len(matrix[0])
# # Create an empty matrix to store the transposed data
# result = [[0 for _ in range(rows)] for _ in range(cols)]
# ​
# for i in range(rows):
# for j in range(cols):
# result[j][i] = matrix[i][j]
# ​
# return result
# ​
# # Input matrix
# matrix = [
# [1, 2, 3],
# [4, 5, 6]
# ]
# ​
# # Transpose the matrix
# transposed_matrix = transpose_matrix(matrix)
# ​
# # Print the transposed matrix
# for row in transposed_matrix:
# print(row)