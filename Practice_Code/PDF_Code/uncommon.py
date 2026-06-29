# Write a Python program to find uncommon words from two Strings.
def uncommon_words(str1, str2):
    words1 = set(str1.split())
    words2 = set(str2.split())
    uncommon_words_set = words1.symmetric_difference(words2)
    uncommon_words_list = ' '.join(uncommon_words_set)
    return uncommon_words_list
string1 = "Python is a programming language"
string2 = "Python is a popular programming language"
result = uncommon_words(string1, string2)
print(f"Uncommon words between the two strings: {result}")
