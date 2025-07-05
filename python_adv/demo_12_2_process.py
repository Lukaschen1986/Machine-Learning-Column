# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings("ignore")
import os
import sys
import numpy as np
import pandas as pd

from concurrent.futures import (ProcessPoolExecutor, as_completed)


# ----------------------------------------------------------------------------------------------------------------------
# path info
path_project = os.getcwd()
print(path_project)

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

def main():
    # 实验-3：多进程
    ## 3.1：submit
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        t0 = pd.Timestamp.now()
        task1 = pool.submit(matrix_multiply, m=20, n=10000, p=40)  # 0 days 00:00:04.864190
        task2 = pool.submit(matrix_multiply, m=30, n=10000, p=50)  # 0 days 00:00:09.039641
        C1 = task1.result()
        C2 = task2.result()
        t1 = pd.Timestamp.now()
        print(f"总耗时：{t1 - t0}")  # 总耗时：0 days 00:00:09.606957

    ## 3.2：as_completed
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        t0 = pd.Timestamp.now()
        list_tasks = [
            pool.submit(matrix_multiply, m=20, n=10000, p=40),  # 0 days 00:00:04.738057
            pool.submit(matrix_multiply, m=30, n=10000, p=50)   # 0 days 00:00:08.787529
        ]
        list_results = [task.result() for task in as_completed(list_tasks)]
        t1 = pd.Timestamp.now()
        print(f"总耗时：{t1 - t0}")  # 总耗时：0 days 00:00:09.346965

    ## 3.3：map
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        t0 = pd.Timestamp.now()
        list_m = [20, 30]
        list_n = [10000, 10000]
        list_p = [40, 50]
        gen_tasks = pool.map(matrix_multiply, list_m, list_n, list_p)
        list_results = [result for result in gen_tasks]  # 0 days 00:00:04.643925, 0 days 00:00:08.586679
        t1 = pd.Timestamp.now()
        print(f"总耗时：{t1 - t0}")  # 总耗时：0 days 00:00:09.130872


if __name__ == "__main__":
    """
    在 Windows 上，multiprocessing 模块要求 所有多进程代码必须在 if __name__ == "__main__": 块中执行，否则会因子进程无法启动而报错。
    """
    main()