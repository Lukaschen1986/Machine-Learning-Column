# -*- coding: utf-8 -*-
"""
pip install tqdm
"""
from time import sleep
from tqdm import (tqdm, trange)


# tqdm
for _ in tqdm(range(10000)):
    sleep(0.001)


# trange
for _ in trange(10000):
    sleep(0.001)
'''
def trange(*args, **kwargs):
    """Shortcut for tqdm(range(*args), **kwargs)."""
    return tqdm(range(*args), **kwargs)
'''

for _ in trange(10000, desc="进度条在这里"):
    sleep(0.001)
    

# use tqdm in generator
epoch = 50
def gen_func():
    for i in range(epoch):
        yield i

for _ in tqdm(gen_func(), total=epoch):
    sleep(0.5)


# 手动添加进度条
pbar = tqdm(total=100)
pbar.update(10)
sleep(2)
pbar.update(20)
sleep(2)
pbar.update(70)
pbar.close()

with tqdm(total=100):
    pbar.update(10)
    sleep(2)
    pbar.update(20)
    sleep(2)
    pbar.update(70)


