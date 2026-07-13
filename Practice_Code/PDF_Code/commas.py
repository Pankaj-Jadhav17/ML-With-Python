# Write a program that accepts a comma separated sequence of words as input and
# prints the words in a comma-separated sequence after sorting them 
input_sequence = input("Enter a comma-separated sequence of words: ")
words = input_sequence.split(',')
sort_words = sorted(words)
sorted_squence = ','.join(sort_words)
print("Sorted sequence of words:", sorted_squence)
