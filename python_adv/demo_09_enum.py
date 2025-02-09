"""
enum 适用场景：当有多个常量需要维护时，enum 提升可读性
"""
from enum import (Enum, IntEnum, StrEnum, auto)


# 不使用 enum
def handel_http(status: int) -> None:
    if status == 200:
        ...
    elif status == 404:
        ...
    return 


# 使用 enum
class HttpStatus(IntEnum):
    OK = 200
    NOT_FOUND = 404

def handel_http(status: HttpStatus) -> None:
    if status == HttpStatus.OK:
        ...
    elif status == HttpStatus.NOT_FOUND:
        ...
    return 

print(HttpStatus.OK)  # 200
handel = handel_http(status=HttpStatus.OK)


# class Color(IntEnum):
#     RED = 1
#     GREEN = 2
#     BLUE = 3
    
#     @classmethod
#     def print_members(cls):
#         for c in cls:
#             print(c.name, c.value)

# Color.print_members()

# class Light(object):
#     def __init__(self, color: Color):
#         self.color = color

# light = Light(Color.RED)