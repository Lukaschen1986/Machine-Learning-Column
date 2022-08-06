# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 11:32:56 2022

@author: lukas

https://zhuanlan.zhihu.com/p/547123604
1. Taichi 是编译性的，而 Python 是解释性的 
2. Taichi 能自动并行，而 Python 通常是单线程的 
3. Taichi 能在 GPU 上运行，而 Python 本身是在 CPU 上运行的
"""

import os
import sys
import warnings; warnings.filterwarnings("ignore")
from functools import wraps
import numpy as np
import pandas as pd
import taichi as ti


ti.init(arch=ti.cpu)
#ti.init(arch=ti.gpu)

# ----------------------------------------------------------------------------------------------------------------
# gedt_time
def get_time(func):
    @wraps(func)
    def inner(*args, **kwargs):
        t0 = pd.Timestamp.now()
        res = func(*args, **kwargs)
        t1 = pd.Timestamp.now()
        print(func.__name__, t1 - t0)
        return res
    return inner


# ----------------------------------------------------------------------------------------------------------------
# 计算素数个数
@ti.func
def is_prime(n: int) -> bool:
    result = True
    for k in range(2, int(n ** 0.5) + 1):
        if n % k == 0:
            result = False
            break
    return result


@get_time
@ti.kernel
def count_primes(n: int) -> int:
    count = 0
    for k in range(2, n):
        if is_prime(k):
            count += 1
    return count


print(count_primes(1000000))  # count_primes 0 days 00:00:03.646250
print(count_primes(1000000))  # count_primes 0 days 00:00:00.138587


# ----------------------------------------------------------------------------------------------------------------
# 动规
N = 15000
f = ti.field(dtype=ti.i32, shape=(N + 1, N + 1))

a_numpy = np.random.randint(0, 100, N, dtype=np.int32)
b_numpy = np.random.randint(0, 100, N, dtype=np.int32)

@get_time
@ti.kernel
def compute_lcs(a: ti.types.ndarray(), b: ti.types.ndarray()) -> ti.i32:
    len_a, len_b = a.shape[0], b.shape[0]

    ti.loop_config(serialize=False)  # False 并行；True 单线程
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            f[i, j] = max(
                    f[i-1, j-1] + (a[i-1] == b[j-1]), 
                    max(f[i-1, j], f[i, j-1])
                    )

    return f[len_a, len_b]


print(compute_lcs(a_numpy, b_numpy))

# ----------------------------------------------------------------------------------------------------------------
# 1143. 最长公共子序列
N = 10000
a = np.random.randint(0, 100, N, dtype=np.int32)
b = np.random.randint(0, 100, N, dtype=np.int32)

@get_time
def longestCommonSubsequence1(a, b) -> int:
    n = len(a)
    p = len(b)
    dp = [[0 for _ in range(p+1)] for _ in range(n+1)]
    
    for i in range(1, n+1):
        for j in range(1, p+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[-1][-1]

longestCommonSubsequence1(a, b)  # 1813 longestCommonSubsequence1 0 days 00:00:54.140244


n = len(a)
p = len(b)
dp = ti.field(dtype=ti.i32, shape=(n+1, p+1))

@get_time
@ti.kernel
def longestCommonSubsequence2(a: ti.types.ndarray(), b: ti.types.ndarray()) -> ti.i32:
    n = a.shape[0]
    p = b.shape[0]
    ti.loop_config(serialize=False)  # False 并行；True 单线程
    
    for i in range(1, n+1):
        for j in range(1, p+1):
            if a[i-1] == b[j-1]:
                dp[i, j] = dp[i-1, j-1] + 1
            else:
                dp[i, j] = ti.max(dp[i-1, j], dp[i, j-1])
    
    return dp[n, p]


longestCommonSubsequence2(a, b)  # 1813 longestCommonSubsequence2 0 days 00:00:00.080785

