# -*- coding: utf-8 -*-
import os
import sys
import copy
from functools import wraps
from joblib import (dump, load)
import numpy as np
import pandas as pd
from concurrent.futures import (ThreadPoolExecutor, ProcessPoolExecutor)
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest



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
# iForest by hand
class TreeNode(object):
    def __init__(self, left=None, right=None, split_attr=None, split_val=None, c_factor=None):
        self.left = left
        self.right = right
        self.split_attr = split_attr
        self.split_val = split_val
        self.c_factor = c_factor
    
    
    def path_length(self, x, current_height=0):
        if (not self.left) and (not self.right):
            return current_height + self.c_factor
        
        if x[self.split_attr] < self.split_val:
            return self.left.path_length(x, current_height + 1)
        else:
            return self.right.path_length(x, current_height + 1)



class iForest(object):
    """
    前提假设：
    1、异常样本占比：不能太多
    2、异常样本取值：尽可能正常样本差异大
    """
    def __init__(self, n_trees, sample_size, threshold):
        self.n_trees = n_trees
        self.sample_size = min(256, sample_size)
        self.threshold = threshold
        self._iforest = []
        self._height_limit = int(np.ceil(np.log2(max(self.sample_size, 2))))
        print(f"sample_size is {self.sample_size}, so height_limit set to {self._height_limit}")
    
    
    @get_time
    def fit(self, x_train, parallel=False):
        if isinstance(x_train, pd.DataFrame):
            x_train = x_train.values
        
        if parallel:
            self.x_train = x_train
            with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
                pool_map = pool.map(self._builid_itree_parallel, range(self.n_trees))
                self._iforest = [task for task in pool_map]
        else:
            for i in range(self.n_trees):
                itree = self._builid_itree_single(x_train)
                self._iforest.append(itree)
        
        return self
    
    
    def _sampling(self, x_train):
        n, _ = x_train.shape
        x_sample = x_train[np.random.choice(n, self.sample_size)]
        return x_sample
    
    
    def _builid_itree(self, x_sample, current_height):
        # 递归终止条件
        """
        1、达到预设的树高度
        2、只剩下一个样本点
        3、剩下的数据全部相同（特例）
        """
        if (current_height >= self._height_limit) or (len(x_sample) <= 1):
            c_factor = iForest._c(len(x_sample))
            return TreeNode(None, None, None, None, c_factor)
        
        # 递归建树
        q = np.random.choice(x_sample.shape[1])
        x_column = x_sample[:, q]
        minv = x_column.min()
        maxv = x_column.max()
        
        if minv == maxv:
            c_factor = iForest._c(len(x_sample))
            return TreeNode(None, None, None, None, c_factor)
        
        p = np.round(np.random.uniform(minv, maxv), 6)
        x_l = x_sample[x_column < p, :]
        x_r = x_sample[x_column >= p, :]
        
        itree = TreeNode()
        itree.split_attr = q
        itree.split_val = p
        
        itree.left = self._builid_itree(x_l, current_height + 1)
        itree.right = self._builid_itree(x_r, current_height + 1)
        return itree
    
    
    def _builid_itree_single(self, x_train):
        x_sample = self._sampling(x_train)
        itree = self._builid_itree(x_sample, 0)
        return itree
    
    
    def _builid_itree_parallel(self, i):
        x_sample = self._sampling(self.x_train)
        itree = self._builid_itree(x_sample, 0)
        return itree
    
    
    @staticmethod
    def _c(n):
        if n < 2:
            return 0
        elif n == 2:
            return 1
        else:
            return np.round(2*(np.log(n-1) + np.euler_gamma) - (2*(n-1)/n), 6)
        
        
    def decision_function(self, x_test):
        avg_path_length = self.get_path_length(x_test)
        c_n = iForest._c(self.sample_size)
        score = 2**(-np.divide(avg_path_length, c_n))
        return score
    
    
    def get_path_length(self, x_test):
        if isinstance(x_test, pd.DataFrame):
            x_test = x_test.values
        
        lst_length = []
        
        for x_i in x_test:
            lst_length_i = []
            
            for itree in self._iforest:
                length = itree.path_length(x_i)
                lst_length_i.append(length)
            
            avg_length_i = np.array(lst_length_i).mean()
            lst_length.append(avg_length_i)
        
        return np.array(lst_length)
    
    
    def predict(self, x_test):
        score = self.decision_function(x_test)
        anomaly = np.where(score >= self.threshold, 1, 0).astype(int)
        return anomaly
    
    

if __name__ == "__main__":
    path_data = os.path.dirname(__file__)
    
#    # get data
#    data = load_iris(as_frame=True)
#    x, y = data.data, data.target
#    N, p = x.shape
#    
#    # 植入随机异常种子
#    contamination = 0.1
#    n_outlier = int(np.ceil(N*contamination))
#    outliers = np.random.uniform(5, 10, [n_outlier, p])
#    
#    for i in range(len(outliers)):
#        row = np.random.choice(x.index)
#        x.iloc[row] += outliers[i]
#    
#    x_train, x_test = train_test_split(x, test_size=0.3, shuffle=True)
#    df_test = copy.deepcopy(x_test)
#    
#    x_train = x_train.values
#    x_test = x_test.values
#    
#    # save data
#    dump(x_train, os.path.join(path_data, "x_train.txt"), compress=9)
#    dump(x_test, os.path.join(path_data, "x_test.txt"), compress=9)
#    dump(df_test, os.path.join(path_data, "df_test.txt"), compress=9)
    
    # load data
    x_train = load(os.path.join(path_data, "x_train.txt"))
    x_test = load(os.path.join(path_data, "x_test.txt"))
    df_test = load(os.path.join(path_data, "df_test.txt"))
        
    # iForest
    model = iForest(n_trees=100, sample_size=len(x_train), threshold=0.6)
    model.fit(x_train, parallel=False)
    
    df_test["score"] = model.decision_function(x_test)
    df_test["anomaly"] = model.predict(x_test)
    
    # IsolationForest
    model = IsolationForest(n_estimators=100,
                            max_samples="auto",
                            contamination=0.1,
                            max_features=1.0,
                            bootstrap=True,
                            n_jobs=-1,
                            random_state=0)
    model.fit(x_train)
    
    df_test["score2"] = model.decision_function(x_test)
    df_test["anomaly2"] = model.predict(x_test)
    
        
