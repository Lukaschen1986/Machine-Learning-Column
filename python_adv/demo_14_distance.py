# -*- coding: utf-8 -*-
import os
import sys
import numpy as np

from geopy.distance import geodesic
from haversine import (haversine, haversine_vector, Unit)
from scipy.spatial import cKDTree


# ----------------------------------------------------------------------------------------------------------------------
# 单点坐标 (纬度, 经度)
point = (39.9042, 116.4074)  # 北京
batch_points = [
    (31.2304, 121.4737),  # 上海
    (23.1291, 113.2644),  # 广州
    (34.0522, 118.2437)   # 洛杉矶（远距离测试）
]

# ----------------------------------------------------------------------------------------------------------------------
# flag
flag = 1

# ----------------------------------------------------------------------------------------------------------------------
# geopy
if flag == 0:
    distances = [geodesic(point, p).km for p in batch_points]

# ----------------------------------------------------------------------------------------------------------------------
# haversine
if flag == 1:
    point = np.array(point)
    batch_points = np.array(batch_points)
    distances = haversine_vector(point, batch_points, unit=Unit.KILOMETERS, comb=True)

# ----------------------------------------------------------------------------------------------------------------------
# scipy.spatial
if flag == 2:
    # 将经纬度转换为弧度（Haversine公式需要）
    batch_rad = np.radians(batch_points)
    tree = cKDTree(batch_rad, metric="haversine")
    
    # 查询单个点（转换为弧度）
    query_rad = np.radians(point)
    distances_rad, index = tree.query(query_rad, k=len(batch_rad))
    
    # 转换为千米（地球半径约6371 km）
    distances = distances_rad[0] * 6371
    print(distances)  # 输出: [1067.68 1887.45 10041.23]
    



