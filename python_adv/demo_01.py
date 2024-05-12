# -*- coding: utf-8 -*-
class Student:
    student_num = 0
    def __init__(self, name, sex):
        self.name = name
        self.sex = sex
        Student.student_num += 1
    
    @classmethod
    def add_students(cls, add_num):
        cls.student_num += add_num

    @classmethod
    def from_string(cls, info):
        name, sex = info.split(' ')
        return cls(name, sex)
    
    @staticmethod
    def name_len(name):
        return len(name)

stu = Student('Qiqi', 'Female')
print(f'Student.student_num:{Student.student_num}')

stu1 = Student.from_string('Qiqi Male')
print(f'stu.name: {stu.name}\nstu.sex: {stu.sex}')
print(f'stu name: {stu.name}, name len: {Student.name_len(stu.name)}')