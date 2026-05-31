# Write a Python Program to calculate your Body Mass Index.
# Body Mass Index (BMI) is a measure of body fat based on an individual's weight and
# height. It is commonly used as a screening tool to categorize individuals into different weight
# status categories, such as underweight, normal weight, overweight, and obesity.
# The BMI is calculated using the following formula:
# (kg)
# BMI = Weight
# Height (m)2
# Alternatively, in the imperial system:
# Weight (lb) × 703
# BMI = Height
# (in)2
# BMI provides a general indication of body fatness but does not directly measure body fat or
# distribution. It is widely used in public health and clinical settings as a quick and simple tool
# to assess potential health risks associated with weight. Different BMI ranges are associated
# with different health categories, but it's important to note that BMI has limitations and does
# not account for factors such as muscle mass or distribution of fat.

def bodymassindex(weight, height):
    return round(weight / (height ** 2), 2)

h = float(input("Enter your height in meters: "))
w = float(input("Enter your weight in kilograms: "))

print("Welcome to the Body Mass Index (BMI) Calculator!")
bmi = bodymassindex(w, h)
print("Your BMI is:", bmi)

if bmi < 18.5:
    print("You are underweight.")
elif 18.5 <= bmi < 24.9:
    print("You have a normal weight.")
elif 25 <= bmi < 29.9:
    print("You are overweight.")
else:
    print("You are obese.")
