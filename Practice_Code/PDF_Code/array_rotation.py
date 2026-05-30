# Write a Python Program for array rotation.

def rotate_array(arr, d):
    n = len(arr)
    if d < 0 or d >= n:
        return "Rotation count must be between 0 and the length of the array - 1"
    rotated_arr = [0] * n
    for i in range(n):
        rotated_arr[i] = arr[(i + d) % n]
    return rotated_arr
arr = [1, 2, 3, 4, 5]
d = 2
result = rotate_array(arr, d)
print("Original array:", arr)
print("Rotated array:", result)