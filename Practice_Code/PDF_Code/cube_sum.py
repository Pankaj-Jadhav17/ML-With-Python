# Write a Python Program for cube sum of first n natural numbers?

def cube_sum_of_natural_numbers(n):
    if n <= 0:
        print("Please enter a positive integer.")
    else:
        Total_sum = (n * (n + 1) // 2) ** 2
        return Total_sum
    
n = int(input("Enter a positive integer: "))
if n <= 0:
    print("Please enter a positive integer.")
else:
    result = cube_sum_of_natural_numbers(n)
    print(f"The cube sum of first {n} natural numbers is: {result}")
    
    