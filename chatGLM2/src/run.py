# -*- coding: utf-8 -*-
import os
import numpy as np
import torch as th
from torch import nn
from transformers import (pipeline, 
                          AutoTokenizer, BertTokenizer,
                          AutoModel, BertModel, BertConfig,
                          AutoModelForSequenceClassification)
from datasets import (load_dataset, load_from_disk)
from torchcrf import CRF
import torch.optim as optim

device = th.device("cuda" if th.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------------------------------------------------
# 路径
path_project = "C:/my_project/MyGit/Machine-Learning-Column/chatGLM2"
path_data = os.path.join(path_project, "data")
path_model = os.path.join(path_project, "model")

# ----------------------------------------------------------------------------------------------------------------
# 加载分词器
checkpoint = "chatglm2-6b-int4"

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=os.path.join(path_model, checkpoint),
    cache_dir=path_model,
    force_download=False,
    local_files_only=True,
    trust_remote_code=True
)

# ----------------------------------------------------------------------------------------------------------------
# 加载预训练模型
model = AutoModel.from_pretrained(
    pretrained_model_name_or_path=os.path.join(path_model, checkpoint),
    cache_dir=path_model,
    force_download=False,
    local_files_only=False,
    trust_remote_code=True
    ).half().cuda()

# for param in model.parameters():
#     param.requires_grad_(False)

response, his = model.chat(tokenizer, query="你好", history=[])
print(response)  







