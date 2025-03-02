# -*- coding: utf-8 -*-
"""
https://support.huaweicloud.com/usermanual-maas-modelarts/maas-modelarts-0042.html
https://support.huaweicloud.com/usermanual-maas-modelarts/maas-modelarts-0026.html
"""
import warnings; warnings.filterwarnings("ignore")
import os
import json
import requests

from dotenv import load_dotenv
from openai import OpenAI


# ----------------------------------------------------------------------------------------------------------------
load_dotenv(dotenv_path="explore.env")
baidu_key = os.getenv("BAIDU_KEY")
zhipu_key = os.getenv("ZHIPU_KEY")
ma_key = os.getenv("MA_KEY")

# ----------------------------------------------------------------------------------------------------------------
# 自定义工具
cityName2districtId = {
    "南京": "320100",
    "深圳": "440300"
}

def get_weather(cityName):
    districtId = cityName2districtId.get(cityName)
    url = f"https://api.map.baidu.com/weather/v1/?district_id={districtId}&data_type=all&ak={baidu_key}"
    response = requests.get(url)
    data = response.json()
    return json.dumps(data)

get_weather_tool = {
    "name": "get_weather",
    "description": "根据输入的城市名称，查询天气",
    "parameters": {
        "type": "object",
        "properties": {
            "cityName": {
                "type": "string",
                "description": "城市名称"
            }
        },
        "required": ["cityName"]
    }
}

tools = [
    {
        "type": "function",
        "function": get_weather_tool
    }
]

tool_dict = {
    "get_weather": get_weather
}

# ----------------------------------------------------------------------------------------------------------------
plan = "1"
# plan = "2"

if plan == "1":
    base_url = "https://infer-modelarts-cn-southwest-2.modelarts-infer.com/v1/infers/4957fe8d-ef53-4c7b-85c9-d1afd88e48ed/v1"
    model = "Qwen2.5-72B-32K"
    api_key = ma_key
elif plan == "2":
    base_url = "https://open.bigmodel.cn/api/paas/v4"
    model = "glm-4-plus"
    api_key = zhipu_key

client = OpenAI(
    api_key=api_key,
    base_url=base_url
)

# ----------------------------------------------------------------------------------------------------------------
def main():
    # prompt
    system_prompt = "You are a helpful assistant on business travel."
    user_prompt = "帮我查下南京的天气"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # 思考模型：第一次请求，调用工具
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        stream=False
    )
    
    assistant_response = response.choices[0].message
    assistant_response.model_dump()
    '''
    {'content': None,
     'refusal': None,
     'role': 'assistant',
     'audio': None,
     'function_call': None,
     'tool_calls': [{'id': 'chatcmpl-tool-08cd2bd278204fde93796b5d25aacaa1',
       'function': {'arguments': '{"cityName": "\\u5357\\u4eac"}',
        'name': 'get_weather'},
       'type': 'function'}]}
    '''
    messages.append(assistant_response)
    
    tool_name = assistant_response.tool_calls[0].function.name
    tool_to_call = tool_dict.get(tool_name)
    tool_args = json.loads(assistant_response.tool_calls[0].function.arguments)
    
    tool_response = tool_to_call(**tool_args)
    print(tool_response)
    '''
    {"status": 0, 
     "result": 
         {"location": 
          {"country": "\u4e2d\u56fd", "province": "\u6c5f\u82cf\u7701", 
           "city": "\u5357\u4eac\u5e02", "name": "\u5357\u4eac", "id": "320100"}, 
          "now": {"text": "\u6674", "temp": 17, "feels_like": 16, "rh": 67, 
                  "wind_class": "3\u7ea7", "wind_dir": "\u4e1c\u5357\u98ce", 
                  "uptime": "20250228214000"}, 
          "forecasts": [{"text_day": "\u591a\u4e91", "text_night": "\u591a\u4e91", 
                         "high": 21, "low": 13, "wc_day": "3~4\u7ea7", "wd_day": "\u4e1c\u98ce", 
                         "wc_night": "<3\u7ea7", "wd_night": "\u4e1c\u5357\u98ce", 
                         "date": "2025-02-28", "week": "\u661f\u671f\u4e94"}, 
                        {"text_day": "\u6674", "text_night": "\u591a\u4e91", "high": 26, 
                         "low": 15, "wc_day": "<3\u7ea7", "wd_day": "\u897f\u98ce", 
                         "wc_night": "<3\u7ea7", "wd_night": "\u4e1c\u5357\u98ce", 
                         "date": "2025-03-01", "week": "\u661f\u671f\u516d"}, 
                        {"text_day": "\u5c0f\u96e8", "text_night": "\u4e2d\u96e8", 
                         "high": 28, "low": 10, "wc_day": "<3\u7ea7", 
                         "wd_day": "\u5357\u98ce", "wc_night": "3~4\u7ea7", 
                         "wd_night": "\u5317\u98ce", "date": "2025-03-02", 
                         "week": "\u661f\u671f\u65e5"}, 
                        ]}, "message": "success"}
    '''
    messages.append(
        {
            "role": "tool",
            "content": tool_response,
            "tool_call_id": assistant_response.tool_calls[0].id
        }
    )

    # 问答模型：第二次请求，整理工具结果
    second_response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False
    )
    print(second_response.choices[0].message.content)
    '''
    南京当前的天气情况如下：
    
    - 天气状况：晴
    - 当前温度：17°C
    - 体感温度：16°C
    - 相对湿度：67%
    - 风力：3级
    - 风向：东南风
    - 更新时间：2025年2月28日 21:40
    
    未来几天的天气预报如下：
    
    - 2月28日（星期五）：白天多云，夜间多云，最高温度21°C，最低温度13°C，白天风力3~4级，风向东风，夜间风力小于3级，风向东南风。
    - 3月1日（星期六）：白天晴，夜间多云，最高温度26°C，最低温度15°C，白天风力小于3级，风向西风，夜间风力小于3级，风向东南风。
    - 3月2日（星期日）：白天小雨，夜间中雨，最高温度28°C，最低温度10°C，白天风力小于3级，风向南风，夜间风力3~4级，风向北风。
    - 3月3日（星期一）：白天中雨，夜间阴，最高温度13°C，最低温度3°C，白天风力小于3级，风向北风，夜间风力小于3级，风向西北风。
    - 3月4日（星期二）：白天阴，夜间多云，最高温度8°C，最低温度3°C，白天风力小于3级，风向西北风，夜间风力小于3级，风向东北风。
    
    请注意根据天气变化适时调整着装，祝您在南京的行程愉快！
    '''
