# Write a Python program to find all duplicate characters in string.
def find_duplicates(s):
    char_count = {}
    duplicates = []
    for char in s:
        char_count[char] = char_count.get(char, 0) + 1
    for char, count in char_count.items():
        if count > 1:
            duplicates.append(char)
    return duplicates

# Example usage
input_string = "programming"
result = find_duplicates(input_string)
print("Duplicate characters:", result)