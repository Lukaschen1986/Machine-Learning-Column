from functools import wraps
import random as rd
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# 构造数据集
n = 100
x0 = np.random.uniform(0, 10, [n, 3])
x1 = np.random.uniform(10, 20, [n, 3])
x = np.concatenate([x0, x1], axis=0)
w = np.random.normal(0, 2, [3, 1])
y = x.dot(w) + np.random.normal(0, 0.1, [2*n, 1])
y = y.reshape(-1)

x = pd.DataFrame(x, columns=["x1", "x2", "x3"])
y = pd.Series(y, name="y")

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=0)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, shuffle=True, random_state=0)

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
# 手写 XGBoost
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


class XGBoost(CART):
    def __init__(self, min_samples_split=2, max_depth=6, n_estimators=100, learning_rate=0.1,
                 alpha=0, lamb=1, gamma=0, subsample=0.9, colsample=0.9):
        CART.__init__(self, min_samples_split, max_depth)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.lamb = lamb
        self.gamma = gamma
        self.subsample = subsample
        self.colsample = colsample
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
    
    
    @get_time
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
            D = self._add_gradient_info(D, y_train_pre)  # 预先计算本轮的 g 和 h写
            self._tree = self._build_tree(D, 0)  # 重写
            self._forest[epoch] = self._tree
            
            y_train_hat = self.predict(x_train)  # 继承
            y_train_pre += self.learning_rate*y_train_hat
            
            # valid
            y_valid_hat = self.predict(x_valid)  # 继承
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
        n, p = x_train.shape
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
    
    
    def _build_tree(self, D, current_depth):
        # 递归终止条件
        if len(D) < self.min_samples_split:
            c = self._get_c(D)
            return TreeNode(None, None, None, None, c)
        if current_depth == self.max_depth:
            c = self._get_c(D)
            return TreeNode(None, None, None, None, c)
        
        # 递归建树
        best_gain = float("-inf")
        
        for attr in D.columns[3:]:
            D = D.sort_values(by=attr)
            
            for i in range(1, len(D)):
                D1 = D[0:i]
                D2 = D[i:]
                g1 = D1.g; h1 = D1.h
                g2 = D2.g; h2 = D2.h
                g = D.g; h = D.h
                gain = self._get_gain(g, h, g1, h1, g2, h2)
                
                if gain > best_gain:
                    best_gain = gain
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
    
    
    def _get_c(self, D):
        c = -1.0 * (np.sum(D.g) + self.alpha) / (np.sum(D.h) + self.lamb)
        return np.round(c, 6)
    
    
    def _get_gain(self, g, h, g1, h1, g2, h2):
        left = (np.sum(g1) + self.alpha)**2 / (np.sum(h1) + self.lamb)
        right = (np.sum(g2) + self.alpha)**2 / (np.sum(h2) + self.lamb)
        parent = (np.sum(g) + self.alpha)**2 / (np.sum(h) + self.lamb)
        gain = left + right - parent - self.gamma
        return gain


    def predict_boosting(self, x_test):
        y_test_res = np.zeros(len(x_test))
        
        for (epoch, sub_tree) in self._forest.items():
            self._tree = sub_tree  # 关键
            y_test_hat = self.predict(x_test)
            y_test_res += self.learning_rate*y_test_hat
        
        return y_test_res



if __name__ == "__main__":
    XGBoost.mro()
    
    # 手写 XGBoost
    estimator = XGBoost(min_samples_split=2, max_depth=6, n_estimators=100, learning_rate=0.1,
                        alpha=0, lamb=1, gamma=0, subsample=0.9, colsample=0.9)
    estimator.fit(x_train, y_train, x_valid, y_valid)
    
    y_hat = estimator.predict_boosting(x_test)
    rmse = mean_squared_error(y_test, y_hat)
    print(rmse)
    
    
