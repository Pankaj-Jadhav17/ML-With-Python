numbers = [5, 10, 3, 8, 2]
largest = numbers[0]
for i in numbers:
    if i > largest:
        largest = i
        print("The largest number in the list is:", largest)
        