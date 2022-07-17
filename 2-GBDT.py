# -*- coding: utf-8 -*-
import os
import sys
import copy
#from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from CART.run import CART


path_project = "D:/my_project/Python_Project/ML/GBDT/"


# 构造数据集
n = 100
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


class GBDT(CART):
    def __init__(self, min_samples_split=2, max_depth=3, n_estimators=100, learning_rate=0.1):
        CART.__init__(self, min_samples_split, max_depth)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self._forest = {}
        self._eval_result = {}
    
    
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
            r_train = y_train - y_train_pre
            D = pd.concat([r_train, x_train], axis=1)
            self._depth = -1
            self._tree = self._build_tree(D)  # 继承 CART
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


    def predict_boost(self, x_test):
        y_test_res = np.zeros(len(x_test))
        
        for (epoch, sub_tree) in self._forest.items():
            self._tree = sub_tree
            y_test_hat = self.predict(x_test)
            y_test_res += self.learning_rate*y_test_hat
        
        return y_test_res
    


if __name__ == "__main__":
    print(path_project)
    
    # sklearn GBDT
    estimator = GradientBoostingRegressor(loss="squared_error", min_samples_split=2, max_depth=3,
                                          n_estimators=100, learning_rate=0.1)
    estimator.fit(x_train, y_train)
    y_hat_sk = estimator.predict(x_test)
    rmse_sk = mean_squared_error(y_test, y_hat_sk)
    print(rmse_sk)
    
    # 手写 GBDT
    estimator = GBDT(min_samples_split=2, max_depth=3, n_estimators=100, learning_rate=0.1)
    estimator.fit(x_train, y_train, x_valid, y_valid)
    
    y_hat = estimator.predict_boost(x_test)
    print(y_hat)
    rmse = mean_squared_error(y_test, y_hat)
    print(rmse)
    
