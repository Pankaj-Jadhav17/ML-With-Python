def is_happy_number(num):
    seen = set()
    while num != 1 and num not in seen:
        seen.add(num)
        num = sum(int(digit) ** 2 for digit in str(num))
    return num == 1

num = int(input("Enter a number: "))
if is_happy_number(num):
    print(f"{num} is a happy number.")
else:
    print(f"{num} is not a happy number.")
