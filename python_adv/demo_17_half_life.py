# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def half_life(M, T, t):
    """
    Parameters
    ----------
    M : float
        初始质量
    T : int
        半衰期
    t : int
        反应时间

    Returns
    -------
    m : float
        剩余质量

    """
    param = np.random.uniform(low=0.5, high=0.7)
    m = M * param**(t/T)
    return m



if __name__ == "__main__":
    M = 0  # 初始情绪价值
    T1 = 60  # 稳定型
    T2 = 30  # 普通型
    T3 = 10  # 匮乏型
    T4 = 2  # 黑洞型
    
    list_days = list(range(365))
    list_m1 = []
    list_m2 = []
    list_m3 = []
    list_m4 = []
    
    for t in list_days:
        if t % 30 == 0:
            M += 5000
        if (t % 120 == 0) and (t != 0):
            M += 10000
        if t == 0:
            m1 = M
            m2 = M
            m3 = M
            m4 = M
            
        m1 = 0.8 * m1 + 0.2 * half_life(M, T1, t)
        m2 = 0.8 * m2 + 0.2 * half_life(M, T2, t)
        m3 = 0.8 * m3 + 0.2 * half_life(M, T3, t)
        m4 = 0.8 * m4 + 0.2 * half_life(M, T4, t)
        
        list_m1.append(m1)
        list_m2.append(m2)
        list_m3.append(m3)
        list_m4.append(m4)
    
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.plot(list_m1)
    plt.plot(list_m2)
    plt.plot(list_m3)
    plt.plot(list_m4)
    plt.title(label="情绪半衰曲线")
    plt.xlabel(xlabel="时间")
    plt.ylabel(ylabel="情绪价值")
    plt.legend(["稳定型", "普通型", "匮乏型", "黑洞型"], loc="best", shadow=True)
    
    

