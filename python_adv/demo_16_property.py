# -*- coding: utf-8 -*-
from typing import (Literal, List, Dict, Tuple, Any, TypeVar)


# ----------------------------------------------------------------------------------------------------------------
# 不用 property
class Square1(object):
    def __init__(self, width: int | float, height: int | float) -> None:
        self.width = width
        self.height = height
        self.area = self.width * self.height

square1: object = Square1(width=1.5, height=1.5)
square1.area  # 2.25

square1.width = 2
square1.area  # 2.25, incorrect

# ----------------------------------------------------------------------------------------------------------------
# 简单 property
class Square2(object):
    def __init__(self, width: int | float, height: int | float) -> None:
        self.width = width
        self.height = height
    
    @property
    def area(self) -> int | float:
        return self.width * self.height

square2: object = Square2(width=1.5, height=1.5)
square2.area  # 2.25

square2.width = 2
square2.area  # 3.0, correct

square2: object = Square2(width=-1.5, height=1.5)
square2.area  # -2.25, invalid

# ----------------------------------------------------------------------------------------------------------------
# 复杂 property
class Square3(object):
    def __init__(self, width: int | float, height: int | float) -> None:
        self.width = width
        self.height = height
    
    @property
    def area(self) -> int | float:
        return self.width * self.height
    
    @property
    def width(self) -> int | float:
        return self._width
    
    @width.setter
    def width(self, val: int | float) -> None:
        if (not isinstance(val, (int, float))) or (val < 0):
            raise ValueError
        else:
            self._width = val 
            
    @width.deleter
    def width(self) -> None:
        del self._width
    
square3: object = Square3(width=1.5, height=1.5)
square3.width = 2
square3.area  # 3.0, correct

square3.width = -2  # ValueError, correct
Square3(width=-2, height=1.5)  # ValueError, correct

# ----------------------------------------------------------------------------------------------------------------
# 用描述符(Descriptor)替代property
class ValidNumber(object):
    def __set_name__(self, cls, name):
        self.name = f"_{name}"
    
    def __get__(self, instance, cls):
        return getattr(instance, self.name)
    
    def __set__(self, instance, val):
        if (not isinstance(val, (int, float))) or (val < 0):
            raise ValueError
        else:
            setattr(instance, self.name, val)
    

class Square4(object):
    width = ValidNumber()  # 实例化描述符类
    height = ValidNumber()
    
    def __init__(self, width: int | float, height: int | float) -> None:
        self.width = width
        self.height = height
    
    @property
    def area(self) -> int | float:
        return self.width * self.height
    
square4: object = Square4(width=1.5, height=1.5)
square4.height = -2  # ValueError, correct
Square4(width=1.5, height=-2)  # ValueError, correct


