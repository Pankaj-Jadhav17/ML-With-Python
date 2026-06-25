# Write a Python program for removing 𝑖𝑡ℎ character from a string.
def remove_ith_char(string, i):
    if i < 0 or i >= len(string):
        return "Index out of range"
    return string[:i] + string[i+1:]

string = "Hello, World!"
i = 7
result = remove_ith_char(string, i)
print(f"Original string: {string}")
print(f"String after removing {i}th character: {result}")
