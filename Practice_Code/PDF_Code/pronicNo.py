# Write a Python program to print all pronic numbers between 1 and 100.
def is_pronic_number(num):
    for n in range(1, int(num**0.5) + 1):
        if n * (n + 1) == num:
            return True
    return False

print("Pronic number between 1 and 100:")
for i in range(1, 100):
        if is_pronic_number(i):
            print(i, end = " | ")
            