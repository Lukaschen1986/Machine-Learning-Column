import os
import sys
import requests
import json

from dotenv import load_dotenv


# ----------------------------------------------------------------------------------------------------------------
# 设置环境变量并获取 baidu key
load_dotenv(dotenv_path="baidu.env")
baidu_key = os.getenv("BAIDU_KEY")

# ----------------------------------------------------------------------------------------------------------------
def get_surround(
    city: str, 
    location: str, 
    query: str
    ):
    """查询某个地点周边（2km）的相关信息。
    
    Args:
        city (str): 城市名称，中文如：北京
        location (str): 地点，如“全季酒店”、“坂田基地”
        query (str): 周边类型，如“美食”、“健身中心”

    Returns:
        str: 返回一个 JSON 字符串，包含周边地点的名称、地址、电话、距离（米）、评分和营业时间等信息。
    """
    url_1 = f"https://api.map.baidu.com/place/v3/suggestion?query={location}&region={city}&region_limit=true&output=json&ak={baidu_key}"
    response_1 = requests.get(url_1)
    data_1 = response_1.json()
    origin = (data_1["results"][0]["location"]["lat"], data_1["results"][0]["location"]["lng"])
    
    url_2 = (
        f"https://api.map.baidu.com/place/v3/around?query={query}&location={origin[0]},{origin[1]}&radius=2000&output=json"
        f"&scope=2&sort_name:distance|sort_rule:1&page_size=30&ak={baidu_key}"
    )
    response_2 = requests.get(url_2)
    data_2 = response_2.json()
    
    results = data_2["results"]
    response = []
    for _dict in results:
        d = {
            "name": _dict.get("name"),  # 名称
            "address": _dict.get("address"),  # 地址
            "telephone": _dict.get("telephone"),  # 电话
            "tag": _dict.get("detail_info").get("tag"),  # 标签
            "label": _dict.get("detail_info").get("label"),  # 标签细分解释，比如停车场标签（地上停车场/地下停车场），知名景区标签（几A级景区），酒店标签（什么类型酒店）等
            "distance": _dict.get("detail_info").get("distance"),  # 距离中心点的距离（米）
            "brand": _dict.get("detail_info").get("brand"),  # poi对应的品牌（如加油站中的『中石油』、『中石化』）
            "price": _dict.get("detail_info").get("price"),  # poi商户的价格
            "overall_rating": _dict.get("detail_info").get("overall_rating"),  # 总体评分
            "taste_rating": _dict.get("detail_info").get("taste_rating"),  # 口味评分
            "service_rating": _dict.get("detail_info").get("service_rating"),  # 服务评分
            "environment_rating": _dict.get("detail_info").get("environment_rating"),  # 环境评分
            "facility_rating": _dict.get("detail_info").get("facility_rating"),  # 星级（设备）评分
            "hygiene_rating": _dict.get("detail_info").get("hygiene_rating"),  # 卫生评分
            "shop_hours": _dict.get("detail_info").get("shop_hours")  # 营业时间
        }
        response.append(d)
    return json.dumps(response, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("请提供城市名称、地点和周边类型作为参数，例如：python get_surround_tool.py 南京 软件大道 川菜")
    else:
        city = sys.argv[1]
        location = sys.argv[2]
        query = sys.argv[3]
        surround_info = get_surround(city, location, query)
        print(surround_info)