# Write a Python program to sort Python Dictionaries by Key or Value.

my_dict = {'apple': 3, 'banana': 1, 'cherry': 2}
sorted_dict_by_key = dict(sorted(my_dict.items()))
print("Dictionary sorted by key:", sorted_dict_by_key)
for key, value in sorted_dict_by_key.items():
    print(f"{key}: {value}")
    