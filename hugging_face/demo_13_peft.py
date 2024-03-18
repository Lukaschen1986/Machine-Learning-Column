# -*- coding: utf-8 -*-
import os
import torch as th
from peft import (LoraConfig, get_peft_model, PeftModel)


device = th.device("cuda" if th.cuda.is_available() else "cpu")
print(device)

# ----------------------------------------------------------------------------------------------------------------
# path
# path_project = "C:/my_project/MyGit/Machine-Learning-Column/hugging_face"
path_project = os.getcwd()
path_data = os.path.join(os.path.dirname(path_project), "data")
path_model = os.path.join(os.path.dirname(path_project), "model")

# ----------------------------------------------------------------------------------------------------------------
# net
net_1 = th.nn.Sequential(
    th.nn.Linear(in_features=10, out_features=10),
    th.nn.ReLU(),
    th.nn.Linear(in_features=10, out_features=2)
)
print(net_1)

config_1 = LoraConfig(target_modules=["0"], r=4)
model_1 = get_peft_model(model=net_1, peft_config=config_1)
print(model_1)

model_1.save_pretrained(save_directory=os.path.join(path_model, "lora_1"))

# ----------------------------------------------------------------------------------------------------------------
# 多适配器
net_2 = th.nn.Sequential(
    th.nn.Linear(in_features=10, out_features=10),
    th.nn.ReLU(),
    th.nn.Linear(in_features=10, out_features=2)
)
config_2 = LoraConfig(target_modules=["2"], r=8)
model_2 = get_peft_model(model=net_2, peft_config=config_2)
model_2.save_pretrained(save_directory=os.path.join(path_model, "lora_2"))

inputs = th.arange(0, 10).reshape(1, 10).float()

net_2 = th.nn.Sequential(
    th.nn.Linear(in_features=10, out_features=10),
    th.nn.ReLU(),
    th.nn.Linear(in_features=10, out_features=2)
)
# from_pretrained
model = PeftModel.from_pretrained(
    model=net_2,
    model_id=os.path.join(path_model, "lora_1"),
    adapter_name="lora_1",
    is_trainable=False
)
model.active_adapter  # 'lora_1'
model(inputs)  # tensor([[-2.2362,  0.7878]])

# load_adapter
model.load_adapter(
    model_id=os.path.join(path_model, "lora_2"),
    adapter_name="lora_2",
    is_trainable=False
)
model.set_adapter("lora_2")  # 切换适配器
model.active_adapter  # 'lora_2'
model(inputs)  # tensor([[ 0.2431, -0.8448]], grad_fn=<AddBackward0>)

# ----------------------------------------------------------------------------------------------------------------
# 禁用适配器
model.set_adapter("lora_1")  # 切换适配器
model.active_adapter  # 'lora_1'

with model.disable_adapter():
    print(model(inputs))
