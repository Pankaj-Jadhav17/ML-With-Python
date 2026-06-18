# # Write a Python program to find smallest number in a list.
# def find_smallest_number(numbers):
#     if not numbers:
#         return None  # Return None if the list is empty
#     smallest_number = numbers[0]
#     for number in numbers:
#         if number < smallest_number:
#             smallest_number = number
#     return smallest_number

numbers = [5, 2, 9, 1, 5, 6]
minimum = numbers[0]
for i in numbers:
    if i < minimum:
        minimum = i
print("The smallest number in the list is:", minimum)