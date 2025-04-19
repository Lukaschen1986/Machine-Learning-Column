from abc import (ABC, abstractclassmethod)


class Course(ABC):
    @staticmethod
    def enroll(self, student):
        print(f"{student} enrolled in Course")

class MathCourse(Course):
    def enroll(self, student):
        print(f"{student} enrolled in MathCourse") 

class ProgrammCourse(Course):
    def enroll(self, student):
        print(f"{student} enrolled in ProgrammCourse") 

if __name__ == "__main__":
    courses = [MathCourse(), ProgrammCourse()]
    for course in courses:
        course.enroll("Alice")