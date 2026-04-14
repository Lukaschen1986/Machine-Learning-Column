import os
import sys
import requests
import json

from dotenv import load_dotenv


# ----------------------------------------------------------------------------------------------------------------
# 设置环境变量并获取 uapis key
load_dotenv(dotenv_path="weather.env")
UAPIS_KEY = os.getenv("UAPIS_KEY")

# ----------------------------------------------------------------------------------------------------------------
def get_weather(
    city: str, 
    extended: bool = True,
    forecast: bool = True,
    indices: bool = True,
    ) -> dict:
    """查询天气，这个接口为你提供精准、实时的天气数据，支持国内和国际城市。

    Args:
        city (str): 城市名称，支持中文（北京）和英文（Tokyo）。可选参数，不传时会尝试 IP 自动定位。
        extended (bool, optional): 扩展气象字段（体感温度、能见度、气压、紫外线、空气质量及污染物分项数据）Defaults to True.
        forecast (bool, optional): 多天预报（最多7天，会额外返回每天的最高温度、最低温度，以及日出日落、风速等详细数据） Defaults to True.
        indices (bool, optional): 18项生活指数（穿衣、紫外线、洗车、运动、花粉等） Defaults to True.

    Returns:
        dict: 返回该地区的实时天气信息。
    """
    url = f"https://uapis.cn/api/v1/misc/weather?city={city}&extended={str(extended).lower()}&forecast={str(forecast).lower()}&indices={str(indices).lower()}"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {UAPIS_KEY}"
    }
    response = requests.get(url, headers=headers)
    return response.json()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("请提供城市名称作为参数，例如：python main.py 北京")
    else:
        city = sys.argv[1]
        weather_info = get_weather(city)
        print(json.dumps(weather_info, indent=2, ensure_ascii=False))