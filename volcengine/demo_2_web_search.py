# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings("ignore")
import os
import json
import requests
import tomllib
import time

from pprint import pp
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from volcenginesdkarkruntime import Ark


# ----------------------------------------------------------------------------------------------------------------
project_path = os.path.dirname(__file__)
print(project_path)

with open(os.path.join(project_path, "config.toml"), "br") as f:
    config = tomllib.load(f)

model = config["doubao"]["model"]
url = config["doubao"]["url"]

# ----------------------------------------------------------------------------------------------------------------
load_dotenv(dotenv_path="vol.env")
vol_key = os.getenv("VOL_KEY")

# ----------------------------------------------------------------------------------------------------------------
client = Ark(
    base_url=url,
    api_key=vol_key
)

# ----------------------------------------------------------------------------------------------------------------
# system_prompt = """you are a helpful assistant."""
system_prompt = """
    角色身份：
    你是一个专业的信息搜索、分析、整理助手，擅长根据用户的提问实现“边想边搜边答”功能。
    
    任务详情：
    1、根据用户的提问进行思考和搜索判断（必须实时输出思考过程）
    2、若问题涉及时效性（如最近半年、近期、最近一段时间）、你的知识盲区、答案不明确时，必须调用web_search
    3、思考时需说明“是否需要搜索”、“为什么搜”、“搜索关键词是什么”
    
    注意事项：
    1、优先使用搜索到的资料，引用格式为`[1] (URL地址)`
    2、结构清晰（用序号、分段），多使用简单易懂的表述
    3、结尾需列出所有参考资料（格式：1. [资料标题](URL)）
    """
system_prompt = system_prompt.replace(" ", "")

# user_prompt = "你好"
user_prompt = "近期AI领域有哪些新闻？"

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
    ]

# ----------------------------------------------------------------------------------------------------------------
tools = [
    {
     "type": "web_search",
     "limit": 2,  # 最多返回10条搜索结果
     "max_keyword": 3,
     # "sources": ["xiaohongshu", "douyin"],
     "user_location": {
          "type": "approximate",
          "country": "中国",
          "region": "江苏",
          "city": "南京"
          }
     }
    ]

# ----------------------------------------------------------------------------------------------------------------
def response_without_stream():
    response = client.responses.create(
        model=model,
        input=messages,
        tools=tools,
        # temperature=0.5,
        # top_p=0.5,
        extra_body={"thinking": {"type": "auto"}},  # auto, disabled, enabled
        stream=False,  # True, False
        )
    
    getattr(response.output[0], "type", "")  # web_search_call
    print(response.output[0])
    '''
    ResponseFunctionWebSearch(
        id='ws_02176085172312100000000000000000000ffffac15bcbd1d884e', 
        action=ActionSearch(query='2025年10月20日南京天气', type='search', sources=None), 
        status='completed', 
        type='web_search_call'
        )
    '''
    
    getattr(response.output[1], "type", "")  # reasoning
    print(response.output[1])
    '''
    ResponseReasoningItem(
        id='rs_02176085172496900000000000000000000ffffac15bcbd99a099', 
        summary=[Summary(text='现在我需要处理用户的问题：“明天南京的天气？”根据当前时间2025年10月19日，
                         用户询问的是10月20日的天气情况。首先，...。
                         因此，可能的结论是明天阴，可能有小雨，温度11-15℃，风力较大。', type='summary_text')], 
                         type='reasoning', 
                         content=None, 
                         encrypted_content=None, 
                         status='completed'
                         )
    '''
    
    getattr(response.output[2], "type", "")  # message
    print(response.output[2])
    '''
    ResponseOutputMessage(
        id='msg_02176085174534500000000000000000000ffffac15bcbd15583c', 
        content=[ResponseOutputText(
            annotations=[
                AnnotationURLCitation(
                    end_index=None, 
                    start_index=None, 
                    title='南京天气预报,南京7天天气预报,南京15天天气预报,南京天气查询', 
                    type='url_citation', 
                    url='http://www.weather.com.cn/weather/101190101.shtml?t=1438673026219', 
                    logo_url='https://p3-search.byteimg.com/img/labis/dafd663cfa3b7fce9addcca7916010cb~noop.jpeg', 
                    site_name='搜索引擎-中国天气网', 
                    publish_time='2025年10月19日 07:30:00(CST) 星期日', 
                    cover_image={'url': 'https://i.tq121.com.cn/i/picList/wf_spring_h.jpg', 'width': 0, 'height': 0}, 
                    summary='7天\n19日（今天）\n小雨转阴...
                    ), 
                AnnotationURLCitation(
                    ...
                    ), 
                AnnotationURLCitation(
                    ...
                    ), 
                ], 
            text='根据搜索结果，南京明天（2025年10月20日）的天气情况存在一定差异，综合权威来源信息整理如下：...',
            type='output_text',
            logprobs=None
            )],
        role='assistant', 
        status='completed', 
        type='message'
        )
    '''
    
    output_text = response.output_text
    print(output_text)
    return 


def response_with_stream():
    response = client.responses.create(
        model=model,
        input=messages,
        tools=tools,
        extra_body={"thinking": {"type": "auto"}},  # auto, disabled, enabled
        stream=True,
        )
    
    thinking_started = False  # AI思考过程是否已开始打印
    answering_started = False  # AI回答是否已开始打印
    
    print("=== 边想边搜启动 ===")
    for chunk in response:  # 遍历每一个实时返回的片段（chunk）
        chunk_type = getattr(chunk, "type", "")  # 获取片段类型（思考/搜索/回答）
        
        # 处理AI思考过程（实时打印“为什么搜、搜什么”）
        if chunk_type == "response.reasoning_summary_text.delta":
            if not thinking_started:
                print(f"\n🤔 AI思考中 [{datetime.now().strftime('%H:%M:%S')}]:")
                thinking_started = True
            # 打印思考内容（delta为实时增量文本）
            print(getattr(chunk, "delta", ""), end="", flush=True)
        
        # 处理搜索状态（开始/完成提示）
        elif "web_search_call" in chunk_type:
            if "in_progress" in chunk_type:
                print(f"\n\n🔍 开始搜索 [{datetime.now().strftime('%H:%M:%S')}]")
            elif "completed" in chunk_type:
                print(f"\n✅ 搜索完成 [{datetime.now().strftime('%H:%M:%S')}]")
        
        # 处理搜索关键词（展示AI实际搜索的内容）
        elif (chunk_type == "response.output_item.done") \
              and hasattr(chunk, "item") \
              and str(getattr(chunk.item, "id", "")).startswith("ws_"):  # ws_为搜索结果标识
                  if hasattr(chunk.item.action, "query"):
                      search_keyword = chunk.item.action.query
                      print(f"\n📝 本次搜索关键词：{search_keyword}")
                      
        # 处理最终回答（实时整合搜索结果并输出）
        elif chunk_type == "response.output_text.delta":
            if not answering_started:
                print(f"\n\n💬 AI回答 [{datetime.now().strftime('%H:%M:%S')}]:")
                print("-" * 50)
                answering_started = True
            # 打印回答内容（实时增量输出）
            print(getattr(chunk, "delta", ""), end="", flush=True)

    # 5. 流程结束
    print(f"\n\n=== 边想边搜完成 [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ===")
    return 
        


if __name__ == "__main__":
    print(f"问：{user_prompt}")
    # response_without_stream()
    response_with_stream()
    
        
    