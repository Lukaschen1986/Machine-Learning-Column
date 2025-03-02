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
ma_key = os.getenv("MA_KEY")
zhipu_key = os.getenv("ZHIPU_KEY")
ds_key = os.getenv("DEEPSEEK_KEY")

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
# plan = "1"
# plan = "2"
plan = "3"

if plan == "1":
    url = "https://infer-modelarts-cn-southwest-2.modelarts-infer.com/v1/infers/4957fe8d-ef53-4c7b-85c9-d1afd88e48ed/v1/chat/completions"
    model = "Qwen2.5-72B-32K"
    Authorization = "Bearer " + ma_key
elif plan == "2":
    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    model = "glm-4-plus"
    Authorization = zhipu_key
elif plan == "3":
    url = "https://api.deepseek.com/chat/completions"
    model = "deepseek-chat"
    Authorization = "Bearer " + ma_key

headers = {
    'Content-Type': 'application/json',
    'Authorization': Authorization
}

# ----------------------------------------------------------------------------------------------------------------
def main():
    # prompt
    system_prompt = "You are a helpful assistant on business travel."
    user_prompt = "帮我查下南京的天气"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    data = {
        "model": model,
        "max_tokens": 256,
        "messages": messages,
        "temperature": 1.0,
        "tools": tools,
        "stream": True,
    }
    
    resp = requests.post(url, headers=headers, json=data, verify=False)
    print(resp.status_code)
    print(resp.text)
    
    for r in resp:
        print(r)
        print("------")
    
    '''
    1 - "stream": False
    1.1 - Qwen2.5-72B-32K
        {
    	"id": "chat-c3eed57068124ca0acaa7bc469601ca8",
    	"object": "chat.completion",
    	"created": 1740796692,
    	"model": "Qwen2.5-72B-32K",
    	"choices": [{
    		"index": 0,
    		"message": {
    			"role": "assistant",
    			"content": null,
    			"tool_calls": [{
    				"id": "chatcmpl-tool-e9c8276cd8a74b4b8913b1d858a31734",
    				"type": "function",
    				"function": {
    					"name": "get_weather",
    					"arguments": "{\"cityName\": \"\\u5357\\u4eac\"}"
    				}
    			}]
    		},
    		"logprobs": null,
    		"finish_reason": "tool_calls",
    		"stop_reason": null
    	}],
    	"usage": {
    		"prompt_tokens": 172,
    		"total_tokens": 193,
    		"completion_tokens": 21
    	},
    	"prompt_logprobs": null
    }
    
    1.2 - glm-4-plus
        {
    	"id": "2025030110430201023ddd0f4646bc",
    	"created": 1740796983,
    	"model": "glm-4-plus",
    	"choices": [{
    		"index": 0,
    		"message": {
    			"role": "assistant",
    			"tool_calls": [{
    				"id": "call_-8929800541279756832",
    				"type": "function",
    				"function": {
    					"name": "get_weather",
    					"arguments": "{\"cityName\": \"南京\"}"
    				},
    				"index": 0
    			}]
    		}
    		"finish_reason": "tool_calls"
    	}],
    	"request_id": "2025030110430201023ddd0f4646bc",
    	"usage": {
    		"completion_tokens": 11,
    		"prompt_tokens": 128,
    		"total_tokens": 139
    	}
    }
    
    2 - "stream": True
    2.1 - Qwen2.5-72B-32K
    data: {"id":"chat-0ce39f45881e4df69b6a5b956e46c738","object":"chat.completion.chunk","created":1740826834,"model":"Qwen2.5-72B-32K","choices":[{"index":0,"delta":{"role":"assistant"},"logprobs":null,"finish_reason":null}]}

    data: {"id":"chat-0ce39f45881e4df69b6a5b956e46c738","object":"chat.completion.chunk","created":1740826834,"model":"Qwen2.5-72B-32K","choices":[{"index":0,"delta":{"tool_calls":[{"id":"chatcmpl-tool-efd231cc9c784398839866c0a9dfd87d","type":"function","index":0}]},"logprobs":null,"finish_reason":null}]}

    data: {"id":"chat-0ce39f45881e4df69b6a5b956e46c738","object":"chat.completion.chunk","created":1740826834,"model":"Qwen2.5-72B-32K","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"name":"get_weather"}}]},"logprobs":null,"finish_reason":null}]}

    data: {"id":"chat-0ce39f45881e4df69b6a5b956e46c738","object":"chat.completion.chunk","created":1740826834,"model":"Qwen2.5-72B-32K","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"cityName\": \""}}]},"logprobs":null,"finish_reason":null}]}

    data: {"id":"chat-0ce39f45881e4df69b6a5b956e46c738","object":"chat.completion.chunk","created":1740826834,"model":"Qwen2.5-72B-32K","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\u5357\\u4eac"}}]},"logprobs":null,"finish_reason":null}]}

    data: {"id":"chat-0ce39f45881e4df69b6a5b956e46c738","object":"chat.completion.chunk","created":1740826834,"model":"Qwen2.5-72B-32K","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":""}}]},"logprobs":null,"finish_reason":null}]}

    data: {"id":"chat-0ce39f45881e4df69b6a5b956e46c738","object":"chat.completion.chunk","created":1740826834,"model":"Qwen2.5-72B-32K","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"}"}}]},"logprobs":null,"finish_reason":null}]}

    data: {"id":"chat-0ce39f45881e4df69b6a5b956e46c738","object":"chat.completion.chunk","created":1740826834,"model":"Qwen2.5-72B-32K","choices":[{"index":0,"delta":{"content":""},"logprobs":null,"finish_reason":"tool_calls","stop_reason":null}]}

    data: [DONE]
    
    2.2 - glm-4-plus
    data: {"id":"2025030119050107f2aa800e544259","created":1740827101,"model":"glm-4-plus","choices":[{"index":0,"finish_reason":"tool_calls","delta":{"role":"assistant","tool_calls":[{"id":"call_2025030119050107f2aa800e544259_0","index":0,"type":"function","function":{"name":"get_weather","arguments":"{\"cityName\": \"南京\"}"}}]}}]}

    data: {"id":"2025030119050107f2aa800e544259","created":1740827101,"model":"glm-4-plus","choices":[{"index":0,"finish_reason":"tool_calls","delta":{"role":"assistant","content":""}}],"usage":{"prompt_tokens":128,"completion_tokens":11,"total_tokens":139}}

    data: [DONE]
    '''
    
    b"\\u5357\\u4eac".decode("unicode-escape")
    