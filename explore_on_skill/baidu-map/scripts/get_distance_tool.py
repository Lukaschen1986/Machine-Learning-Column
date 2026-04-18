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
def get_distance(
    city: str, 
    from_location: str, 
    end_location: str
    ) -> str:
    """查询两个地点之间的距离和预计时间。
    
    Args:
        city (str): 城市名称，中文如：北京
        from_location (str): 出发地点
        end_location (str): 到达地点

    Returns:
        str: 返回一个 JSON 字符串，包含驾车、骑行、步行和公交四种交通方式的距离(米)和预计时间（分钟）。
    """
    # 获取出发地点和到达地点的经纬度
    url_a = f"https://api.map.baidu.com/place/v2/suggestion?query={from_location}&region={city}&city_limit=true&output=json&ak={baidu_key}"
    response_a = requests.get(url_a)
    data_a = response_a.json()
    origin = (data_a["result"][0]["location"]["lat"], data_a["result"][0]["location"]["lng"])
    
    url_b = f"https://api.map.baidu.com/place/v2/suggestion?query={end_location}&region={city}&city_limit=true&output=json&ak={baidu_key}"
    response_b = requests.get(url_b)
    data_b = response_b.json()
    destination = (data_b["result"][0]["location"]["lat"], data_b["result"][0]["location"]["lng"])
    
    # 查询驾车路线的距离和预计时间
    url_driving = f"https://api.map.baidu.com/direction/v2/driving?origin={origin[0]},{origin[1]}&destination={destination[0]},{destination[1]}&ak={baidu_key}"
    response_driving = requests.get(url_driving)
    data_driving = response_driving.json()
    distance_driving = data_driving["result"]["routes"][0]["distance"]
    duration_driving = data_driving["result"]["routes"][0]["duration"]
    
    # 查询骑行路线的距离和预计时间
    url_riding = f"https://api.map.baidu.com/direction/v2/riding?origin={origin[0]},{origin[1]}&destination={destination[0]},{destination[1]}&ak={baidu_key}"
    response_riding = requests.get(url_riding)
    data_riding = response_riding.json()
    distance_riding = data_riding["result"]["routes"][0]["distance"]
    duration_riding = data_riding["result"]["routes"][0]["duration"]
    
    # 查询步行路线的距离和预计时间
    url_walking = f"https://api.map.baidu.com/direction/v2/walking?origin={origin[0]},{origin[1]}&destination={destination[0]},{destination[1]}&ak={baidu_key}"
    response_walking = requests.get(url_walking)
    data_walking = response_walking.json()
    distance_walking = data_walking["result"]["routes"][0]["distance"]
    duration_walking = data_walking["result"]["routes"][0]["duration"]
    
    # 查询公交路线的距离和预计时间
    url_transit = f"https://api.map.baidu.com/direction/v2/transit?origin={origin[0]},{origin[1]}&destination={destination[0]},{destination[1]}&ak={baidu_key}"
    response_transit = requests.get(url_transit)
    data_transit = response_transit.json()
    distance_transit = data_transit["result"]["routes"][0]["distance"]
    duration_transit = data_transit["result"]["routes"][0]["duration"]
    
    result = {
        "driving": {"distance": distance_driving, "duration": duration_driving // 60},
        "riding": {"distance": distance_riding, "duration": duration_riding // 60},
        "walking": {"distance": distance_walking, "duration": duration_walking // 60},
        "transit": {"distance": distance_transit, "duration": duration_transit // 60}
    }
    return json.dumps(result, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("请提供城市名称、出发地点和到达地点作为参数，例如：python get_distance_tool.py 北京 天安门 故宫")
    else:
        city = sys.argv[1]
        from_location = sys.argv[2]
        end_location = sys.argv[3]
        distance_info = get_distance(city, from_location, end_location)
        print(distance_info)