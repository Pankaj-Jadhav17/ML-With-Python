# Write a Python program to Remove empty List from List.
list_of_lists = [1, 2, [], 3, [], 4, 5, [], 6]
filtered_list = [i for i in list_of_lists if i ]
print("The list after removing empty lists is:", filtered_list)