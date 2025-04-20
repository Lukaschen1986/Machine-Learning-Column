from collections import namedtuple
from typing import NamedTuple

# 常规方法
p1 = {"x": 1, "y": 2}
p2 = {"x": 2, "y": 3}

# collections.nametuple 方法
Point = namedtuple(typename="Point", field_names=["x", "y"])
p1 = Point(x=1, y=2)
print(p1.x)  # 1
print(type(p1.x))

# typing.NamedTuple 方法
# class Point(NamedTuple):
#     x: int
#     y: int

# p1 = Point(x=1, y=2)
# print(p1.x)  # 1