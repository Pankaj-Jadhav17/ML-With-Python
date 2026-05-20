# Write a Python Program to Find LCM.
# Least Common Multiple (LCM):
# LCM, or Least Common Multiple, is the smallest multiple that is exactly divisible by two or
# more numbers.
# Formula:
# For two numbers a and b, the LCM can be found using the formula:
# |𝑎 ⋅ 𝑏|
# LCM(𝑎,𝑏) = GCD(𝑎,𝑏)
# For more than two numbers, you can find the LCM step by step, taking the LCM of pairs of
# numbers at a time until you reach the last pair.
# Note: GCD stands for Greatest Common Divisor, which can be calculated using the Euclidean algorithm.

# Function to compute LCM
def compute_lcm(x, y):

    # choose the greater number
    if x > y:
        greater = x
    else:
        greater = y

    while True:
        if (greater % x == 0) and (greater % y == 0):
            lcm = greater
            break

        greater += 1

    return lcm


# Take input from user
num1 = int(input("Enter first number: "))
num2 = int(input("Enter second number: "))

print("The L.C.M. is", compute_lcm(num1, num2))