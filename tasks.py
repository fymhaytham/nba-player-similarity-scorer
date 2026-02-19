# day 1 task a
num1 = int(input("number 1: "))
num = int(input("number 2: "))

print(num1 * num)

# day 1 task b
celcius = float(input("enter celcius: "))
fahrenheit = float((celcius * 9/5) + 32)

print(f"Cº: {celcius} Fº: {fahrenheit}")

# day 2 task a
score = int(input("Enter score: "))
if score < 0 or score > 100:
    print("Invalid score. Must be between 0-100")
elif score >=40:
    print("Pass")
else:
    print("Fail")

# day 2 task b
year = int(input("Enter year:"))
if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
    print("Leap year")
else:
    print("not a leap year")

# day 3 task a
n = int(input("enter number: "))
sum = 0
for i in range(n+1): # range(start, stop) -> integers from start to stop-1
    sum += i

print(sum)

# day 3 task b
n = int(input("enter number: "))
for i in range(n):
    print('*' * i)

# day 4 task a
def max3(num, num1, num2):
    return max(num, num1, num2)

max3(3,5,8)

# day 4 task b
def sum_to_n(n):
    sum = (n*(n+1))/2
    return sum

sum_to_n(5)

# day 5 task a
tasks = {}

def add_task(name):
    tasks[name] = "Incomplete"
def list_tasks():
    return tasks.keys()

task = input("enter task to add: ")
add_task(task)
print(list_tasks())

# day 5 task b
def dict_out(nums):
    stats = {
        "min": min(nums),
        "max": max(nums),
        "avg": sum(nums) / len(nums)
    }
    return stats

# day 6 task a
names = ["Stephen", "Lebron", "Kevin", "Anthony", "Luka", "Giannis", "Anthony"]
print(sorted(set(names)))

# day 6 task b
string = set("Stephen".lower())
string1 = set("James".lower())

print(string & string1)