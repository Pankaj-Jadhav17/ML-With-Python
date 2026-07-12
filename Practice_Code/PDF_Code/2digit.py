# Write a program which takes 2 digits, X,Y as input and generates a 2-dimensional
# array. The element value in the i-th row and j-th column of the array should be i*j.

X, Y = map(int, input("Enter two digits (X, Y): ").split(','))
array = [[0 for j in range(Y)] for i in range(X)]
for i in range(X):
    for j in range(Y):
        array[i][j] = i * j
# print("Generated 2-dimensional array:")
for row in array:
    print(row)
    
    