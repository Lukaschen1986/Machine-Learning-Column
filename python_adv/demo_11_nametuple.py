from collections import namedtuple
from typing import NamedTuple


# 常规方法
class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

# typing.NamedTuple 方法
class Point(NamedTuple):
    x: int
    y: int

# collections.nametuple 方法
Point = namedtuple(typename="Point", field_names=["x", "y"])

p1 = Point(x=1, y=2)
print(p1.x)  # 1