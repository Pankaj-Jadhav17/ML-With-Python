# Write a Python program to split and join a string.
input_str = "Python program to split and join a string."
word_list = input_str.split()
separator = "-"
output_str = separator.join(word_list)

print(f"Original string: {input_str}")
print(f"List of words: {word_list}")
print(f"String after joining words with '{separator}': {output_str}")