import os
from functools import wraps
import random as rd
import numpy as np
import pandas as pd
from concurrent.futures import (ThreadPoolExecutor, ProcessPoolExecutor)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# 构造数据集
n = 50
x0 = np.random.uniform(0, 10, [n, 5])
x1 = np.random.uniform(10, 20, [n, 5])
x = np.concatenate([x0, x1], axis=0)
w = np.random.normal(0, 2, [5, 1])
y = x.dot(w) + np.random.normal(0, 0.1, [2*n, 1])
y = y.reshape(-1)

x = pd.DataFrame(x, columns=["x1", "x2", "x3", "x4", "x5"])
y = pd.Series(y, name="y")

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=0)


# ----------------------------------------------------------------------------------------------------------------
# get_time
def get_time(func):
    @wraps(func)
    def inner(*arg, **kwargs):
        t0 = pd.Timestamp.now()
        res = func(*arg, **kwargs)
        t1 = pd.Timestamp.now()
        print(func.__name__, t1 - t0)
        return res
    return inner


# ----------------------------------------------------------------------------------------------------------------
# 手写 RandomForest
class TreeNode(object):
    def __init__(self, left=None, right=None, split_attr=None, split_val=None, c=None):
        self.left = left
        self.right = right
        self.split_attr = split_attr
        self.split_val = split_val
        self.c = c
    
    
    def get_c(self, x, current_depth=0):
        if (not self.left) and (not self.right):
            return self.c
        
        if x[self.split_attr] < self.split_val:
            return self.left.get_c(x, current_depth + 1)
        else:
            return self.right.get_c(x, current_depth + 1)



class RandomForest(CART):
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, max_features="sqrt"):
        CART.__init__(self, min_samples_split, max_depth)
        self.n_estimators = n_estimators
        self.max_features = max_features
        self._forest = []
    
    
    def check(self):
        if not isinstance(self.min_samples_split, int):
            raise TypeError("min_samples_split must be an integer")
        if self.min_samples_split < 2:
            raise ValueError("min_samples_split must greater than 1")
        if self.max_depth <= 0:
            raise ValueError("max_depth must be greater than zero")
        if not isinstance(self.n_estimators, int):
            raise TypeError("n_estimators must be an integer") 
        if self.n_estimators < 1:
            raise ValueError(f"n_estimators must greater than 0 but was {self.n_estimators}")
        if self.max_features not in {"sqrt", "all"}:
            raise ValueError("max_features must be 'sqrt' or 'all'")
        
    
    @get_time
    def fit(self, x_train, y_train, parallel=False):
        # 参数校验
        self.check()
        
        # 并行
        if parallel:
            self.x_train = x_train
            self.y_train = y_train
            with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
                pool_map = pool.map(self._build_tree_parallel, range(self.n_estimators))
                self._forest = [task for task in pool_map]
        # 串行
        else:
            for epoch in range(self.n_estimators):
                print(f"epoch = {epoch}")
                tree = self._build_tree_single(x_train, y_train)
                self._forest.append(tree)
        
        return self
    
    
    def _build_tree_single(self, x_train, y_train):
        D = self._sampling(x_train, y_train)
        tree = self._build_tree(D, 0)
        return tree
    
    
    def _build_tree_parallel(self, i):
        print(f"epoch = {i}")
        D = self._sampling(self.x_train, self.y_train)
        tree = self._build_tree(D, 0)
        return tree
    
    
    def _sampling(self, x_train, y_train):
        if self.max_features == "all":
            D = pd.concat([y_train, x_train], axis=1)
        elif self.max_features == "sqrt":
            _, p = x_train.shape
            col = rd.sample(x_train.columns.tolist(), int(np.sqrt(p)))
            x_sub = x_train.loc[:, col]
            D = pd.concat([y_train, x_sub], axis=1)
        else:
            raise ValueError("max_features must be 'sqrt' or 'all'")
        return D
    
    
    def predict_bagging(self, x_test):
        y_test_res = np.zeros(len(x_test))
        
        for sub_tree in self._forest:
            self._tree = sub_tree  # 关键
            y_test_hat = self.predict(x_test)
            y_test_res += y_test_hat
        
        y_test_res /= self.n_estimators
        return y_test_res



if __name__ == "__main__":
    # 手写 RandomForestRegressor
    estimator = RandomForest(n_estimators=10, max_depth=None, min_samples_split=2, max_features="sqrt")
    estimator.fit(x_train, y_train)
#    estimator.fit(x_train, y_train, parallel=True)
    
    y_hat = estimator.predict_bagging(x_test)
    rmse = mean_squared_error(y_test, y_hat)
    print(rmse)
    
    # sklearn RandomForestRegressor
    estimator = RandomForestRegressor(n_estimators=10, criterion="squared_error", 
                                      max_depth=None, min_samples_split=2,
                                      max_features="sqrt", n_jobs=-1,
                                      random_state=0)
    estimator.fit(x_train, y_train)
    
    y_hat = estimator.predict(x_test)
    rmse = mean_squared_error(y_test, y_hat)
    print(rmse)
