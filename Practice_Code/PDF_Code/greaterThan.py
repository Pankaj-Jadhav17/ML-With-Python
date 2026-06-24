# Write a Python program to find words which are greater than given length k.
def find_words(word, k):
    result = []
    for i in word:
        if len(i) > k:
            result.append(i)
    return result

word_list = ["hello", "world", "Python", "programming", "code"]
k = 5
long_words = find_words(word_list, k)
print(f"Words greater than length {k} are: {long_words}")

