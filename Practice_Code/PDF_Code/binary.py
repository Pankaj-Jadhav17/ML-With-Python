# Write a Python program to check if a given string is binary string or not.

def is_binary_string(s):
    for char in s:
        if char not in ['0', '1']:
            return False
    return True
input_string = "1010101"
if is_binary_string(input_string):
    print(f"{input_string} is a binary string.")
        

        