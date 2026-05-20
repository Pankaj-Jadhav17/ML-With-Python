# Write a Python Program to Find HCF.
# Highest Common Factor(HCF):
# HCF, or Highest Common Factor, is the largest positive integer that divides two or more
# numbers without leaving a remainder.
# Formula:
    
# For two numbers a and b, the HCF can be found using the formula:
# HCF(𝑎,𝑏) = GCD(𝑎,𝑏)

# For more than two numbers, you can find the HCF by taking the GCD of pairs of numbers at
# a time until you reach the last pair.
# Note: GCD stands for Greatest Common Divisor.

def compute_hcf(x, y):

    if x < y:
        smaller = x
    else:
        smaller = y
    for i in range(1, smaller + 1):
        if (x % i == 0) and (y % i == 0):
            hcf = i
    return hcf
num1 = int(input("Enter first number: "))
num2 = int(input("Enter second number: "))
print("The H.C.F. is", compute_hcf(num1, num2))