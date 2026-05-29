# Write a Python Program to find largest element in an array.

def find_largest_element(arr):
    if not arr:
        return "Array is empty"

    largest = arr[0]

    for num in arr:
        if num > largest:
            largest = num  # Update largest if current number is greater

    return largest  
my_array = [3, 5, 7, 2, 8]
largest_element = find_largest_element(my_array)
print("The largest element in the array is:", largest_element)