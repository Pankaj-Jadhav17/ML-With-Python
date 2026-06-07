# Write a Python program to check if the given number is a Disarium Number.

def is_disarium(n):
    num_str = str(n)
    digit_sum = sum(int(num_str[i]) ** (i + 1) for i in range(len(num_str)))
    return digit_sum == n
try:
    number = int(input("Enter a number: "))
    if is_disarium(number):
        print(f"{number} is a Disarium Number.")
    else:
        print(f"{number} is not a Disarium Number.")
except ValueError:
    print("Please enter a valid integer.")