# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings("ignore")
import os
import sys
import numpy as np
import pandas as pd

from concurrent.futures import (ThreadPoolExecutor, as_completed)


# ----------------------------------------------------------------------------------------------------------------------
# path info
path_project = os.getcwd()
print(path_project)

# ----------------------------------------------------------------------------------------------------------------------
# init data
m = 20
n = 10000
p = 40
A = np.random.uniform(low=0, high=10, size=[m, n])
B = np.random.uniform(low=10, high=20, size=[n, p])

# ----------------------------------------------------------------------------------------------------------------------
# dot
t0 = pd.Timestamp.now()
C = np.dot(A, B)
t1 = pd.Timestamp.now()
print(f"matrix multiply with dot: {t1 - t0}")

# ----------------------------------------------------------------------------------------------------------------------
# for
t0 = pd.Timestamp.now()
C = np.zeros([m, p])

for i in range(m):
    for j in range(p):
        for k in range(n):
            C[i, j] += A[i, k] * B[k, j]

t1 = pd.Timestamp.now()
print(f"matrix multiply with for-loop: {t1 - t0}")

# ----------------------------------------------------------------------------------------------------------------------
# concurrent
def matrix_multiply(m: int, n: int, p: int) -> np.ndarray:
    """矩阵乘法

    Args:
        m (int): A矩阵的行
        n (int): A矩阵的列，同时也是B矩阵的行
        p (int): B矩阵的列

    Returns:
        np.ndarray: C矩阵，m行p列
    """
    A = np.random.uniform(low=0, high=10, size=[m, n])
    B = np.random.uniform(low=10, high=20, size=[n, p])
    C = np.zeros([m, p])
    
    t0 = pd.Timestamp.now()
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    t1 = pd.Timestamp.now()
    print(f"{t1 - t0}")
    return C

# 实验-1：串行
t0 = pd.Timestamp.now()
C1 = matrix_multiply(m=20, n=10000, p=40)  # 0 days 00:00:04.929147
C2 = matrix_multiply(m=30, n=10000, p=50)  # 0 days 00:00:08.982784
t1 = pd.Timestamp.now()
print(f"总耗时：{t1 - t0}")  # 总耗时：0 days 00:00:14.075566

# 实验-2：多线程
## 2.1：submit
with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
    t0 = pd.Timestamp.now()
    task1 = pool.submit(matrix_multiply, m=20, n=10000, p=40)  # 0 days 00:00:11.532392
    task2 = pool.submit(matrix_multiply, m=30, n=10000, p=50)  # 0 days 00:00:16.043308
    C1 = task1.result()
    C2 = task2.result()
    t1 = pd.Timestamp.now()
    print(f"总耗时：{t1 - t0}")  # 总耗时：0 days 00:00:16.088904

## 2.2：as_completed
with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
    t0 = pd.Timestamp.now()
    list_tasks = [
        pool.submit(matrix_multiply, m=20, n=10000, p=40),  # 0 days 00:00:10.759547
        pool.submit(matrix_multiply, m=30, n=10000, p=50)   # 0 days 00:00:14.966385
    ]
    list_results = [task.result() for task in as_completed(list_tasks)]
    t1 = pd.Timestamp.now()
    print(f"总耗时：{t1 - t0}")  # 总耗时：0 days 00:00:15.013062

## 2.3：map
with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
    t0 = pd.Timestamp.now()
    list_m = [20, 30]
    list_n = [10000, 10000]
    list_p = [40, 50]
    gen_tasks = pool.map(matrix_multiply, list_m, list_n, list_p)
    list_results = [result for result in gen_tasks]  # 0 days 00:00:10.235664, 0 days 00:00:14.462606
    t1 = pd.Timestamp.now()
    print(f"总耗时：{t1 - t0}")  # 总耗时：0 days 00:00:14.469606

