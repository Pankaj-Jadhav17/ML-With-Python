# Write a Python program to print all disarium numbers between 1 to 100.

def is_disarium(num):
    str_num = str(num)
    sum_of_digits = sum(int(digit) ** (index + 1) for index, digit in enumerate(str_num))
    return sum_of_digits == num

for i in range(1, 101):
    if is_disarium(i):
        print(i)