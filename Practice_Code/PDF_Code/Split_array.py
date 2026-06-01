def split_and_add(arr, k):
    if k <= 0 or k >= len(arr):
        return arr
    first_part = arr[:k]
    second_part = arr[k:]
    result = second_part + first_part
    return result

arr = [1, 2, 3, 4, 5]
k = 2
result = split_and_add(arr, k)
print("Original array:", arr)
print("Array after splitting and adding:", result)