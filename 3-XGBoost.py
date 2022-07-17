# -*- coding: utf-8 -*-
import os
import sys
import copy
#from collections import defaultdict
import random as rd
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
#from sklearn.ensemble import GradientBoostingRegressor
from GBDT.run import GBDT


path_project = "D:/my_project/Python_Project/ML/XGBoost/"


# 构造数据集
n = 50
x0 = np.random.uniform(0, 10, [n, 4])
x1 = np.random.uniform(50, 60, [n, 4])
x = np.concatenate([x0, x1], axis=0)
w = np.random.uniform(0, 2, [4, 1])
y = x.dot(w) + np.random.normal(0, 0.1, [2*n, 1])
y = y.reshape(-1)

x = pd.DataFrame(x, columns=["x1", "x2", "x3", "x4"])
y = pd.Series(y, name="y")

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=0)


class XGBoost(GBDT):
    def __init__(self, min_samples_split=2, max_depth=6, n_estimators=100, learning_rate=0.1,
                 alpha=0, lamb=1, gamma=0, subsample=0.9, colsample=0.9):
        GBDT.__init__(self, min_samples_split, max_depth, n_estimators, learning_rate)
        self.alpha = alpha
        self.lamb = lamb
        self.gamma = gamma
        self.subsample = subsample
        self.colsample = colsample
    
    
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
        if not isinstance(self.learning_rate, float):
            raise TypeError("learning_rate must be an float") 
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must greater than 0 but was {self.learning_rate}")
        if self.alpha < 0:
            raise ValueError("alpha must greater than 0")
        if self.lamb < 0:
            raise ValueError("lambda must greater than 0")
        if self.gamma < 0:
            raise ValueError("gamma must greater than 0")
        if not isinstance(self.subsample, float):
            raise TypeError("subsample must be an float")
        if (self.subsample <= 0) or (self.subsample > 1):
            raise ValueError("subsample must in (0, 1]")
        if not isinstance(self.colsample, float):
            raise TypeError("colsample must be an float")
        if (self.colsample <= 0) or (self.colsample > 1):
            raise ValueError("colsample must in (0, 1]")
    
    
    def fit(self, x_train, y_train, x_valid, y_valid):
        # 参数校验
        self.check()
        
        # 初始化
        y_train_pre = pd.Series(np.zeros_like(y_train), name="y", index=y_train.index)
        y_valid_res = pd.Series(np.zeros_like(y_valid), name="y", index=y_valid.index)
        self._eval_result["rmse-train"] = []
        self._eval_result["rmse-valid"] = []
        
        # 前向分步
        for epoch in range(self.n_estimators):
            # train
            D = self._sampling(x_train, y_train)  # 行抽样 + 列抽样
            D = self._add_gradient_info(D, y_train_pre)  # 预先计算本轮的 g 和 h
            self._depth = -1  # 初始化树的深度
            self._tree = self._build_tree_xgb(D)  # 重写
            self._forest[epoch] = self._tree
            
            y_train_hat = self.predict(x_train)  # 继承 CART
            y_train_pre += self.learning_rate*y_train_hat
            
            # valid
            y_valid_hat = self.predict(x_valid)  # 继承 CART
            y_valid_res += self.learning_rate*y_valid_hat
            
            # eval
            rmse_train = mean_squared_error(y_train, y_train_pre)
            rmse_valid = mean_squared_error(y_valid, y_valid_res)
            rmse_moving = np.mean(self._eval_result["rmse-valid"][-self.n_estimators//3: ])
            
            if rmse_valid > rmse_moving:
                print(f"iteration stop, best epoch = {epoch}")
                break
            else:
                self._eval_result["rmse-train"].append(rmse_train)
                self._eval_result["rmse-valid"].append(rmse_valid)
            
            # log
            print(f"epoch {epoch}  rmse-train {rmse_train:.4f}  rmse-valid {rmse_valid:.4f}")
        
        return self


    def _sampling(self, x_train, y_train):
        n = len(x_train)
        p = len(x_train.columns)
        
        row = rd.sample(x_train.index.tolist(), int(n*self.subsample))
        col = rd.sample(x_train.columns.tolist(), int(p*self.colsample))
        
        x_sub = x_train.loc[x_train.index.isin(row), col]
        y_sub = y_train.loc[y_train.index.isin(row)]
        D = pd.concat([y_sub, x_sub], axis=1)
        return D
    
    
    def _add_gradient_info(self, D, y_train_pre):
        y_train_pre_sub = y_train_pre.loc[y_train_pre.index.isin(D.index)]
        g = pd.Series(y_train_pre_sub - D.y, name="g", index=D.index)
        h = pd.Series(np.ones_like(D.y), name="h", index=D.index)
        D = pd.concat([g, h, D], axis=1)
        return D
    
    
    def _build_tree_xgb(self, D):
        # 递归终止条件
        if len(D) < self.min_samples_split:
            c = self._get_c(D)
            return c
        if self._depth == self.max_depth:
            c = self._get_c(D)
            return c
        
        # 递归建树
        best_gain = float("-inf")
        
        for col in D.columns[3:]:
            D = D.sort_values(by=col)
            
            for i in range(1, len(D)):
                D1 = D[0: i]
                D2 = D[i: ]
                g1 = D1.g; h1 = D1.h
                g2 = D2.g; h2 = D2.h
                g = D.g; h = D.h
                gain = self._get_gain(g, h, g1, h1, g2, h2)
                
                if gain > best_gain:
                    best_gain = gain
                    best_col = col
                    best_val = np.round(D[col].iloc[i], 6)
                    best_D1 = D1
                    best_D2 = D2
        
        self._depth += 1
        left_rule = " < " + str(best_val)
        right_rule = " >= " + str(best_val)
        node = {best_col: {}}
        node[best_col][left_rule] = self._build_tree_xgb(best_D1)
        node[best_col][right_rule] = self._build_tree_xgb(best_D2)
        return node
    
    
    def _get_c(self, D):
        c = -1.0 * (np.sum(D.g) + self.alpha) / (np.sum(D.h) + self.lamb)
        return np.round(c, 6)
    
    
    def _get_gain(self, g, h, g1, h1, g2, h2):
        left = (np.sum(g1) + self.alpha)**2 / (np.sum(h1) + self.lamb)
        right = (np.sum(g2) + self.alpha)**2 / (np.sum(h2) + self.lamb)
        parent = (np.sum(g) + self.alpha)**2 / (np.sum(h) + self.lamb)
        gain = left + right - parent - self.gamma
        return gain
    
    
    


if __name__ == "__main__":
    print(path_project)
    XGBoost.mro()
    
    # 手写 XGBoost
    estimator = XGBoost(min_samples_split=2, max_depth=6, n_estimators=100, learning_rate=0.1,
                        alpha=0, lamb=1, gamma=0, subsample=0.9, colsample=0.9)
    estimator.fit(x_train, y_train, x_valid, y_valid)
    
    y_hat = estimator.predict_boost(x_test)
    print(y_hat)
    rmse = mean_squared_error(y_test, y_hat)
    print(rmse)
    
