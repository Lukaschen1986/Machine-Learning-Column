# -*- coding: utf-8 -*-
numbers = [1, 2, 3, 4, 5]
first, *rest = numbers
print(first)  # Output: 1
print(rest)  # Output: [2, 3, 4, 5]


# 打包
def print_values(*args):
    print(type(args))  # <class 'tuple'>
    for arg in args:
        print(arg)
print_values(1, 2, 3, 4)  # Output: 10

def example(**kwargs):
    print(type(kwargs))  # <class 'dict'>
    for (k, v) in kwargs.items():
        print(f"{k} = {v}")
example(a=1, b=2, c=3)  # Output: a = 1, b = 2, c = 3


# 解包
def greet(name, age):
    print(f"Hello {name}, you are {age} years old.")

person = ("Alice", 30)
greet(*person)  # Output: Hello Alice, you are 30 years old.

list1 = [1, 2, 3]
tuple1 = (4, 5, 6)
merged = [*list1, *tuple1]  # Result: [1, 2, 3, 4, 5, 6]


def create_profile(name, age, email, profession):
    print(f"Name: {name}")
    print(f"Age: {age}")
    print(f"Email: {email}")
    print(f"Profession: {profession}")

options = {
    "name": "John Doe",
    "age": 30,
    "email": "john.doe@example.com",
    "profession": "Engineer"
}
create_profile(**options)

dict1 = {'a': 1, 'b': 2}
dict2 = {'b': 3, 'c': 4}
merged = {**dict1, **dict2}
print(merged)  # Output: {'a': 1, 'b': 3, 'c': 4}