# -*- coding: utf-8 -*-
from functools import wraps
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.tree import (DecisionTreeClassifier, export_text)


# 构造数据集
n = 100

x0 = np.random.uniform(0, 10, [n, 3])
x1 = np.random.uniform(8, 15, [n, 3])
x = np.concatenate([x0, x1], axis=0)

y0 = np.zeros(n)
y1 = np.ones(n)
y = np.concatenate([y0, y1], axis=0)

x = pd.DataFrame(x, columns=["x1", "x2", "x3"])
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
# 手写 CART
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



class CART(object):
    """
    self = CART(min_samples_split=2, max_depth=3)
    """
    def __init__(self, min_samples_split=2, max_depth=None):
        self.min_samples_split = min_samples_split
        self.max_depth = (float("inf") if not max_depth else max_depth)
        self._tree = None
    
    
    def check(self):
        if not isinstance(self.min_samples_split, int):
            raise TypeError("min_samples_split must be an integer")
        if self.min_samples_split < 2:
            raise ValueError("min_samples_split must greater than 1")
        if self.max_depth <= 0:
            raise ValueError("max_depth must be greater than zero")
    
    
    @get_time
    def fit(self, x_train, y_train):
        # 参数校验
        self.check()
        # 合并数据集
        D = pd.concat([y_train, x_train], axis=1)
        # 构造二叉回归树
        self._tree = self._build_tree(D, 0)
        return self
    
    
    def _build_tree(self, D, current_depth):
        # 递归终止条件
        if len(D) < self.min_samples_split:
            c = np.round(np.mean(D.y), 6)
            return TreeNode(None, None, None, None, c)
        if current_depth == self.max_depth:
            c = np.round(np.mean(D.y), 6)
            return TreeNode(None, None, None, None, c)
        
        # 递归建树
        best_loss = float("inf")
        
        for attr in D.columns[1:]:
            D = D.sort_values(by=attr)
            
            for i in range(1, len(D)):
                D1 = D[0:i]
                D2 = D[i:]
                c1 = np.mean(D1.y)
                c2 = np.mean(D2.y)
                loss1 = CART._get_log_loss(D1.y, c1)
                loss2 = CART._get_log_loss(D2.y, c2)
                loss = loss1 + loss2
                
                if loss < best_loss:
                    best_loss = loss
                    best_attr = attr
                    best_val = np.round(D[attr].iloc[i], 6)
                    best_D1 = D1
                    best_D2 = D2
        
        tree = TreeNode()
        tree.split_attr = best_attr
        tree.split_val = best_val
        tree.left = self._build_tree(best_D1, current_depth + 1)
        tree.right = self._build_tree(best_D2, current_depth + 1)
        return tree
    
    
    @staticmethod
    def _get_log_loss(y_true,  y_hat):
        eps = 10**-8
        res = -1.0 * np.mean(y_true*np.log(y_hat+eps) + (1-y_true)*np.log(1-y_hat+eps))
        return res
    

    def predict_proba(self, x_test):
        n = len(x_test)
        y_hat = np.array([])
        
        for i in range(n):
            x_i = x_test.iloc[i]
            c = self._tree.get_c(x_i)
            y_hat = np.append(y_hat, c)
        
        return y_hat
    


if __name__ == "__main__":
    # 手写 CART
    estimator = CART(min_samples_split=2, max_depth=None)
    estimator.fit(x_train, y_train)
    y_hat = estimator.predict_proba(x_test)
    loss = log_loss(y_test, y_hat)
    print(loss)
    
    # sklearn CART
    estimator = DecisionTreeClassifier(criterion="gini", min_samples_split=2)
    estimator.fit(x_train, y_train)
    y_hat_sk = estimator.predict_proba(x_test)[:,1]
    loss_sk = log_loss(y_test, y_hat_sk)
    print(loss_sk)
    r = export_text(estimator, feature_names=x.columns.tolist())
    print(r)
    
    