# Write a Python program to check order of character in string using OrderedDict().
from collections import OrderedDict

def check_order(string, reference):
    string_dict = OrderedDict.fromkeys(string)
    reference_dict  = OrderedDict.fromkeys(reference)
    return string_dict == reference_dict

input_string = "abc"
reference_string = "abc"
if check_order(input_string, reference_string):
    print("The order of characters in the string is the same as the reference.")
else:
    print("The order of characters in the string is different from the reference.")
    