import os

from typing import Dict
from bilibili_api import (search, sync)


def general_search(keyword: str) -> Dict:
    """使用关键词搜索bilibili-api

    Args:
        keyword (str): 关键词

    Returns:
        Dict: 搜索后返回的结果信息
    """
    info = search.search(keyword)
    return sync(info)


if __name__ == "__main__":
    keyword = "AI"
    result = general_search(keyword)
    print(result)