numbers = [5, 10, -45, 8, 2]
largest = numbers[0]
for i in numbers:
    if i > largest:
        largest = i
        print("The largest number in the list is:", largest)
        