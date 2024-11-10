# -*- coding: utf-8 -*-
from time import sleep
import pandas as pd
import asyncio as aio
import aiohttp
import aiofiles


# ----------------------------------------------------------------------------------------------------------------
# sync
# def fetch_url(url):
#     print("fetching the url")
#     sleep(1)
#     print("finished fetching")
#     return "url_content"

# def read_file(file_path):
#     print("reading the url")
#     sleep(1)
#     print("finished reading")
#     return "file_content"

# def main():
#     url = "example.com"
#     file_path = "example.txt"
#     fetch_result = fetch_url(url)
#     read_result = read_file(file_path)
#     return 

# ----------------------------------------------------------------------------------------------------------------
# async
"""
1、定义协程函数
2、包装协程为任务
3、建立事件循环
"""
async def fetch_url(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()
    # print("fetching the url")
    # await aio.sleep(1)
    # print("finished fetching")
    # return "url_content"

async def read_file(file_path):
    async with aiofiles.open(file_path, "r") as f:
        return await f.read()
    # print("reading the url")
    # await aio.sleep(1)
    # print("finished reading")
    # return "file_content"

async def main():
    url = "https://www.baidu.com"
    file_path = "example.txt"
    
    # 手动
    print("手动")
    task_1 = aio.create_task(fetch_url(url))
    task_2 = aio.create_task(read_file(file_path))
    fetch_result = await task_1
    read_result = await task_2
    
    # 自动-1
    print("自动-1")
    results = aio.gather(fetch_url(url), read_file(file_path))
    print(await results)
    
    # 自动-2
    print("自动-2")
    results = aio.as_completed([fetch_url(url), read_file(file_path)])
    for res in results:
        print(await res)
    return 



if __name__ == "__main__":
    t0 = pd.Timestamp.now()
    # main()  # sync
    aio.run(main())  # async
    t1 = pd.Timestamp.now()
    print(t1 - t0)