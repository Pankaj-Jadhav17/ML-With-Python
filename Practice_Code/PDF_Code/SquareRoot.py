# Write a program that calculates and prints the value according to the given formula:

import math

C = 50
H = 30

def calculate_value(D):
    return math.sqrt((2 * C * D) / H)
input_sequence = input("Enter a sequence of numbers separated by commas: ")
D_values = input_sequence.split(',')
result = [calculate_value(int(D)) for D in D_values]
print(','.join(map(str, result)))
