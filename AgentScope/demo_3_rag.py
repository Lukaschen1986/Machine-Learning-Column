# -*- coding: utf-8 -*-
"""
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple agentscope --pre --user
https://github.com/modelscope/agentscope/blob/main/examples/conversation_with_RAG_agents/rag_example.py
https://blog.csdn.net/Attitude93/article/details/139448187
https://blog.csdn.net/Attitude93/article/details/139452945
https://blog.csdn.net/Attitude93/article/details/139478271
"""
import warnings; warnings.filterwarnings("ignore")
import os
# import sys
# from typing import Any
import json
# import numpy as np
# import pandas as pd
import torch as th
from langchain.document_loaders import TextLoader
from langchain.text_splitter import (CharacterTextSplitter, RecursiveCharacterTextSplitter)
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain import (PromptTemplate, FewShotPromptTemplate)
import agentscope
# from agentscope import msghub
from agentscope.agents import (UserAgent, DialogAgent, DictDialogAgent, ReActAgent, LlamaIndexAgent)
from agentscope.message import Msg
# from agentscope.parsers.json_object_parser import MarkdownJsonDictParser
# from agentscope.service import (ServiceExecStatus, ServiceToolkit, ServiceResponse, 
#                                 bing_search, create_file, download_from_url)
# from agentscope.rag import KnowledgeBank


device = th.device("cuda" if th.cuda.is_available() else "cpu")
devive_cnt = th.cuda.device_count()
print(f"device = {device}; devive_cnt = {devive_cnt}")
print(th.__version__)
print(th.version.cuda)
print(agentscope.__version__)

# ----------------------------------------------------------------------------------------------------------------
# path
path_project = "C:/my_project/MyGit/Machine-Learning-Column/AgentScope"
path_data = os.path.join(os.path.dirname(path_project), "data")
path_model = os.path.join(os.path.dirname(path_project), "model")

# ----------------------------------------------------------------------------------------------------------------
# Use Embedding Model
file_name = "XX大学2024年差旅费报销答疑.txt"
loader = TextLoader(file_path=os.path.join(path_data, file_name), encoding="utf-8")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n"],
    chunk_size=200,
    chunk_overlap=50,
    length_function=len,
    add_start_index=True
)
documents = text_splitter.split_documents(documents=docs)

checkpoint = "m3e-base"
embedding_model = HuggingFaceEmbeddings(
    model_name=os.path.join(path_model, "sentence-transformers", checkpoint),
    cache_folder=os.path.join(path_model, "sentence-transformers", checkpoint),
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True}
    )
db_1 = FAISS.from_documents(documents=documents, embedding=embedding_model)
# db_2 = ...

# ----------------------------------------------------------------------------------------------------------------
# Use DialogAgent
agents = agentscope.init(
    model_configs="./configs/model_configs.json",
    agent_configs="./configs/agent_configs.json",
    project="企业政策问题"
)
rag_agent_qwen = agents[-2]
rag_agent_glm = agents[-1]

query = "网约车发票能不能报销？"
query = "etc怎么报销？"
'''
TO DO 1
根据用户的query和db的名称与描述，构造prompt输入一个路由agent，输出最合适的db
TO DO 2
或者，遍历所有db，找出最合适的片段
此处默认使用db_1
'''
res_similarity = db_1.similarity_search(query, k=3)
context = "\n".join(res.page_content for res in res_similarity)

# QA-Prompt
template = "已知信息如下：\n{context}\n根据已知信息回答问题：\n{query}"
prompt = PromptTemplate.from_template(template)
msg_content = prompt.format(context=context, query=query)

# Inference
msg = Msg(name="user", content=msg_content, role="user")

res_qwen = rag_agent_qwen(msg)
"""
query = "网约车发票能不能报销？"
RAG-Agent-1: 网约车发票不可以报销。根据已知信息，由于网络约车平台如滴滴、易到和神州专车开具的票据在当前缺乏相关政策法规的约束，因此不符合报销条件。
"""
res_glm = rag_agent_glm(msg)
"""
query = "网约车发票能不能报销？"
RAG-Agent-2: 滴滴、易到、神州专车等网络约车平台开具的票据不可以报销，因为目前对于网络约车没有相关政策法规约束。
"""

# ----------------------------------------------------------------------------------------------------------------
# Use LlamaIndexAgent
with open("./configs/agent_configs.json", "r", encoding="utf-8") as f:
    agent_configs = json.load(f)
rag_agent = LlamaIndexAgent(**agent_configs[-3]["args"])

query = "网约车发票能不能报销？"
msg = Msg(name="user", content=query, role="user")
res = rag_agent(msg)
'''ValueError: max() arg is an empty sequence
'''


