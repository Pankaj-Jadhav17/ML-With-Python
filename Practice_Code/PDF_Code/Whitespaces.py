input_sequence = input("Enter a sequence of characters: ")
words = input_sequence.split()
sorted_words = sorted(words)

result = ' '.join(sorted_words)
print("Sorted sequence of characters:", result)
