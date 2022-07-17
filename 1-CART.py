# -*- coding: utf-8 -*-
import os
import sys
import copy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import (DecisionTreeRegressor, export_text)


path_project = "D:/my_project/Python_Project/ML/CART/"


# 构造数据集
n = 50
x0 = np.random.uniform(0, 10, [n, 4])
x1 = np.random.uniform(10, 20, [n, 4])
x = np.concatenate([x0, x1], axis=0)
w = np.random.normal(0, 2, [4, 1])
y = x.dot(w) + np.random.normal(0, 0.1, [2*n, 1])
#y0 = np.zeros([n, 1])
#y1 = np.ones([n, 1])
#y = np.concatenate([y0, y1], axis=0)
y = y.reshape(-1)

x = pd.DataFrame(x, columns=["x1", "x2", "x3", "x4"])
y = pd.Series(y, name="y")

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# 手写 CART
class CART(object):
    """
    self = CART(min_samples_split=2, max_depth=None)
    """
    def __init__(self, min_samples_split=2, max_depth=None):
        self.min_samples_split = min_samples_split
        self.max_depth = (float("inf") if not max_depth else max_depth)
        self._tree = {}
        self._depth = -1
    
    
    def check(self):
        if not isinstance(self.min_samples_split, int):
            raise TypeError("min_samples_split must be an integer")
        if self.min_samples_split < 2:
            raise ValueError("min_samples_split must greater than 1")
        if self.max_depth <= 0:
            raise ValueError("max_depth must be greater than zero")
    
    
    def fit(self, x_train, y_train):
        # 参数校验
        self.check()
        # 合并数据集
        D = pd.concat([y_train, x_train], axis=1)
        # 构造二叉回归树
        self._tree = self._build_tree(D)
        return self
    
    
    def _build_tree(self, D):
        # 递归终止条件
        if len(D) < self.min_samples_split:
            c = np.round(D.y.mean(), 6)
            return c
        if self._depth == self.max_depth:
            c = np.round(D.y.mean(), 6)
            return c
        
        # 递归建树
        best_loss = float("inf")
        
        for col in D.columns[1:]:
            D = D.sort_values(by=col)
            
            for i in range(1, len(D)):
                D1 = D[0: i]
                D2 = D[i: ]
                c1 = D1.y.mean()
                c2 = D2.y.mean()
                loss = np.mean((D1.y - c1)**2) + np.mean((D2.y - c2)**2)
                
                if loss < best_loss:
                    best_loss = loss
                    best_col = col
                    best_val = np.round(D[col].iloc[i], 6)
                    best_D1 = D1
                    best_D2 = D2
        
        self._depth += 1
        left_rule = " < " + str(best_val)
        right_rule = " >= " + str(best_val)
        node = {best_col: {}}
        node[best_col][left_rule] = self._build_tree(best_D1)
        node[best_col][right_rule] = self._build_tree(best_D2)
        return node
    
    
    def predict(self, x_test):
        n = len(x_test)
        array_hat = np.array([])
        recursive_tree = copy.deepcopy(self._tree)
        
        for i in range(n):
            x_test_single = x_test.iloc[i]
            y_hat = self._inference(recursive_tree, x_test_single)
            array_hat = np.append(array_hat, y_hat)
        
        return array_hat
    
    
    def _inference(self, recursive_tree, x_test_single):
        for (col, branch) in recursive_tree.items():
            val = x_test_single[col]
            left_rule = list(branch.keys())[0]
            
            if eval(str(val) + left_rule):
                recursive_tree = list(branch.values())[0]
            else:
                recursive_tree = list(branch.values())[1]
                
            if isinstance(recursive_tree, float):
                return recursive_tree
            else:
                recursive_tree = self._inference(recursive_tree, x_test_single)
                
        return recursive_tree





if __name__ == "__main__":
    print(path_project)
    
    # sklearn CART
    estimator = DecisionTreeRegressor(criterion="squared_error", min_samples_split=2)
    estimator.fit(x_train, y_train)
    y_hat_sk = estimator.predict(x_test)
    print(y_hat_sk)
    rmse_sk = mean_squared_error(y_test, y_hat_sk)
    print(rmse_sk)
    
    r = export_text(estimator, feature_names=x.columns.tolist())
    print(r)
    
    # 手写 CART
    estimator = CART(min_samples_split=2, max_depth=3)
    estimator.fit(x_train, y_train)
    y_hat = estimator.predict(x_test)
    print(y_hat)
    rmse = mean_squared_error(y_test, y_hat)
    print(rmse)
    
    
