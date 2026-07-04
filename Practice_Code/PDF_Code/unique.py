# Write a Python program to Extract Unique dictionary values.
my_dir = {'a': 1, 'b': 2, 'c': 1, 'd': 3}
unique_values = set()

for i in my_dir.values():
    unique_values.add(i)
    unique_list = list(unique_values)
    print("Unique values in the dictionary:", unique_list)