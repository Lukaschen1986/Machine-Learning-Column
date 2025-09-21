# -*- coding: utf-8 -*-
from __future__ import annotations
from datetime import datetime
from tokenize import Name
from typing import (Literal, List, Dict, Tuple, Any, TypeVar)
from collections.abc import (Iterable, Callable)


class Student(object):
    T = TypeVar("T")            # Can be anything
    A = TypeVar("A", str, int)  # Must be str or int
    
    x: int = 10
    
    def __init__(
        self, 
        name: str, 
        birth: datetime | str, 
        sex: Literal["male", "female"], 
        courses: List[str], 
        scores: Dict[str, float], 
        location: Tuple[float, float] | None
        ) -> None:
        self.name = name
        self.birth = birth
        self.sex = sex
        self.courses = courses
        self.scores = scores
        self.location = location

    def follow(self, other_student: Student):  # from __future__ import annotations
        pass

    def print_names(self, names: Iterable[str]) -> List[str]:
        """
        设计理念
        输入要宽松：尽可能接收通用的类型，如 Iterable
        输出要明确：尽可能输出专用的类型，如 List
        """
        lst = []
        for name in names:
            lst.append(name)
        return lst

    def apply_func(
        self, 
        func1: Callable[[str, int], Tuple[str, str]],  # Callable[输入类型, 输出类型]
        func2: Callable[..., Any],  # 多个类型的入参，任意类型的出参
        s1: str,
        n1: int
        ) -> Tuple[str, str]:
        """Callable用法
        
        Args:
            func1 (Callable[[str, int], Tuple[str, str]]): _description_
            n1 (int): _description_

        Returns:
            Tuple[str, str]: _description_
        """
        return func1(s1, n1)
    
    def add(self, a: A, b: A) -> A:
        """泛型

        Args:
            a (A): _description_
            b (A): _description_

        Returns:
            A: _description_
        """
        return a + b





