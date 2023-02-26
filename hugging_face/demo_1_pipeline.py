# -*- coding: utf-8 -*-
'''
https://huggingface.co/course/zh-CN/chapter1/3?fw=pt
'''
import os
import numpy as np
import torch as th
from torch import nn
from transformers import (pipeline, BertTokenizer, AutoTokenizer, BertModel, AdamW)
from datasets import (load_dataset, load_from_disk)
from torchcrf import CRF
import torch.optim as optim

device = th.device("cuda" if th.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------------------------------------------------
# 路径
path_project = "C:/my_project/MyGit/Machine-Learning-Column/hugging_face"
path_data = os.path.join(os.path.dirname(path_project), "data")
path_model = os.path.join(os.path.dirname(path_project), "model")

# ----------------------------------------------------------------------------------------------------------------
# pipeline
# windows默认下载路径 C:\Users\lukas\.cache\huggingface\hub
'''
目前可用的一些pipeline是：
特征提取（获取文本的向量表示）
填充空缺
ner（命名实体识别）
问答
情感分析
文本摘要
文本生成
翻译
零样本分类(对尚未标记的文本进行分类)
'''

# ----------------------------------------------------------------------------------------------------------------
# 情感分析
model = pipeline(task="sentiment-analysis", model=None)
model(sequences="I've been waiting for a HuggingFace course my whole life.")
'''
[{'label': 'POSITIVE', 'score': 0.9598049521446228}]
'''

model(sequences=["I've been waiting for a HuggingFace course my whole life.", 
                 "I hate this so much!"])
'''
[{'label': 'POSITIVE', 'score': 0.9598049521446228},
 {'label': 'NEGATIVE', 'score': 0.9994558691978455}]
'''

# ----------------------------------------------------------------------------------------------------------------
# 零样本分类
model = pipeline(task="zero-shot-classification", model=None)
model(sequences="This is a course about the Transformers library",
      candidate_labels=["education", "politics", "business"])
'''
{'sequence': 'This is a course about the Transformers library',
 'labels': ['education', 'business', 'politics'],
 'scores': [0.8445993065834045, 0.11197393387556076, 0.043426718562841415]}
'''

# ----------------------------------------------------------------------------------------------------------------
# 文本生成
model = pipeline(task="text-generation", model=None)
model = pipeline(task="text-generation", model="distilgpt2")

model(text_inputs="In this course, we will teach you how to")
model(text_inputs="In this course, we will teach you how to",
      num_return_sequences=2,  # 控制生成多少个不同的序列
      max_length=50  # 控制输出文本的总长度
      )

# ----------------------------------------------------------------------------------------------------------------
# 问答
model = pipeline(task="question-answering", model=None)
model(context="My name is Sylvain and I work at Hugging Face in Brooklyn",
      question="Where do I work?")
'''
{'score': 0.6949772238731384, 'start': 33, 'end': 45, 'answer': 'Hugging Face'}
请注意，此pipeline通过从提供的上下文中提取信息；它不会凭空生成答案
'''

# ----------------------------------------------------------------------------------------------------------------
# NER
model = pipeline(task="ner", model=None)
model("My name is Sylvain and I work at Hugging Face in Brooklyn.")

# ----------------------------------------------------------------------------------------------------------------
# 文本摘要
model = pipeline(task="summarization", model=None)
model(text_inputs=
      """
        America has changed dramatically during recent years. Not only has the number of 
        graduates in traditional engineering disciplines such as mechanical, civil, 
        electrical, chemical, and aeronautical engineering declined, but in most of 
        the premier American universities engineering curricula now concentrate on 
        and encourage largely the study of engineering science. As a result, there 
        are declining offerings in engineering subjects dealing with infrastructure, 
        the environment, and related issues, and greater concentration on high 
        technology subjects, largely supporting increasingly complex scientific 
        developments. While the latter is important, it should not be at the expense 
        of more traditional engineering.
    
        Rapidly developing economies such as China and India, as well as other 
        industrial countries in Europe and Asia, continue to encourage and advance 
        the teaching of engineering. Both China and India, respectively, graduate 
        six and eight times as many traditional engineers as does the United States. 
        Other industrial countries at minimum maintain their output, while America 
        suffers an increasingly serious decline in the number of engineering graduates 
        and a lack of well-educated engineers.
      """,
      min_length=30,
      max_length=300
      )

# ----------------------------------------------------------------------------------------------------------------
# 翻译
model = pipeline(task="translation", model="Helsinki-NLP/opus-mt-fr-en")
model("Ce cours est produit par Hugging Face.")

