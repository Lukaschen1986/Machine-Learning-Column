# -*- coding: utf-8 -*-
import os
import sys
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter
# from datasets import (load_dataset, load_from_disk, Dataset)
# from transformers import (AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig,
#                           TrainingArguments, DataCollatorWithPadding, DataCollatorForLanguageModeling,
#                           DataCollatorForSeq2Seq, DataCollatorForTokenClassification)
# from transformers.integrations import TensorBoardCallback
# from peft import (LoraConfig, get_peft_model, PeftModel, TaskType, get_peft_model_state_dict)
# from trl import SFTTrainer


# ----------------------------------------------------------------------------------------------------------------
device = th.device("cuda" if th.cuda.is_available() else "cpu")
devive_cnt = th.cuda.device_count()
print(f"device = {device}; devive_cnt = {devive_cnt}")
print(th.__version__)
print(th.version.cuda)

# ----------------------------------------------------------------------------------------------------------------
path_project = os.getcwd()
path_data = os.path.join(os.path.dirname(path_project), "data")
path_model = os.path.join(os.path.dirname(path_project), "model")
path_output = os.path.join(os.path.dirname(path_project), "output")

# ----------------------------------------------------------------------------------------------------------------
class MoeLayer(nn.Module):
    """
    self = MoeLayer(num_experts, in_features, out_features)
    """
    def __init__(self, num_experts, in_features, out_features):
        super(MoeLayer, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([nn.Linear(in_features, out_features) for _ in range(num_experts)])  # 思考：MoE + LoRA
        self.gate = nn.Linear(in_features, num_experts)
    
    def forward(self, x):
        experts_outputs = th.stack([module(x) for module in self.experts], dim=1)
        gate_score = F.softmax(self.gate(x), dim=1)
        output = th.bmm(gate_score.unsqueeze(1), experts_outputs).squeeze(1)
        return output


if __name__ == "__main__":
    num_experts = 4
    in_features = 4
    out_features = 3
    batch_size = 2
    
    model = MoeLayer(num_experts, in_features, out_features)
    x = th.randn(batch_size, in_features)
    output = model(x)
    
    # th.stack([module(x) for module in self.experts], dim=1)
    # experts_outputs.shape  # torch.Size([2, 4, 3])
    # gate_score.shape
    # gate_score.unsqueeze(1).shape  # torch.Size([2, 1, 4])
    # th.bmm(gate_score.unsqueeze(1), experts_outputs).shape
