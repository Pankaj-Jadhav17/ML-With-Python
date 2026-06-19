n = int(input("Enter the number of elements: "))

numbers = []
for i in range(n):
    num = int(input(f"Enter element {i+1}: "))
    numbers.append(num)

# Find second largest
largest = second_largest = float('-inf')

for i in numbers:
    if i > largest:
        second_largest = largest
        largest = i
    elif largest > i > second_largest:
        second_largest = i

if second_largest == float('-inf'):
    print("There is no second largest number in the list.")
else:
    print("The second largest number in the list is:", second_largest)