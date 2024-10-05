# -*- coding: utf-8 -*-
"""
# !pip install -U accelerate transformers tokenizers deepspeed
# https://www.bilibili.com/video/BV1ZZ421T7XJ?p=1&vd_source=fac9279bd4e33309b405d472b24286a8
# https://huggingface.co/docs/accelerate/basic_tutorials/install
# https://huggingface.co/docs/accelerate/usage_guides/deepspeed
"""
import os
import sys
import warnings; warnings.filterwarnings("ignore")
import torch as th
import torch.nn as nn
import torch.optim as optim
from accelerate import (Accelerator, DeepSpeedPlugin)


# ----------------------------------------------------------------------------------------------------------------
# device = th.device("cuda" if th.cuda.is_available() else "cpu")
accelerator = Accelerator()
device = accelerator.device
devive_cnt = th.cuda.device_count()
print(f"device = {device}; devive_cnt = {devive_cnt}")
print(th.__version__)
print(th.version.cuda)

# ----------------------------------------------------------------------------------------------------------------
path_project = "C:/my_project/MyGit/Machine-Learning-Column/hugging_face"
path_data = os.path.join(os.path.dirname(path_project), "data")
path_model = os.path.join(os.path.dirname(path_project), "model")
path_output = os.path.join(os.path.dirname(path_project), "output")

# ----------------------------------------------------------------------------------------------------------------
# step-1: 配置文件'
'''
复制：python -c "from accelerate.utils import write_basic_config; write_basic_config(mixed_precision='fp16')"
编辑：C:/Users/lukas/.cache/huggingface/accelerate/default_config.yaml
查看：accelerate env
检查：accelerate test

{
  "compute_environment": "LOCAL_MACHINE",
  "debug": false,
  "distributed_type": "MULTI_GPU",  # DEEPSPEED
  "downcast_bf16": false,
  "enable_cpu_affinity": false,
  "machine_rank": 0,
  "main_training_function": "main",
  "mixed_precision": "fp16",
  "num_machines": 1,
  "num_processes": 2,
  "rdzv_backend": "static",
  "same_network": false,
  "tpu_use_cluster": false,
  "tpu_use_sudo": false,
  "use_cpu": false
}
'''

# ----------------------------------------------------------------------------------------------------------------
# step-2: 模型测试
class SimpleNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = th.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    input_dim = 10
    hidden_dim = 20
    output_dim = 2
    batch_size = 64
    data_size = 10000
    
    input_data = th.randn([data_size, input_dim])
    labels = th.randn([data_size, output_dim])
    dataset = th.utils.data.TensorDataset(input_data, labels)
    loader = th.utils.data.DataLoader(dataset=dataset,
                                      batch_size=batch_size,
                                      # collate_fn=collate_fn,
                                      shuffle=True,
                                      drop_last=True)
    
    model = SimpleNet(input_dim, hidden_dim, output_dim).to(device)
    deepspeed = DeepSpeedPlugin(zero_stage=2, gradient_clipping=1.0)
    accelerator = Accelerator(deepspeed_plugin=deepspeed)
    
    opti = optim.AdamW(params=model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=10**-8, weight_decay=0.01)
    objt = nn.MSELoss(reduction="mean")
    model, opti, loader = accelerator.prepare(model, opti, loader)
    
    epochs = 200
    for epoch in range(epochs):
        # train
        loss_tmp = 0
        model.train()
        for (i, (inputs, labels)) in enumerate(loader):
            # inputs = inputs.to(device)
            # labels = labels.to(device)
            
            output_mlp = model(inputs)
            loss = objt(output_mlp, labels)
            loss_tmp += loss.item()
            
            opti.zero_grad()
            # loss.backward()
            accelerator.backward(loss)
            opti.step()
        
        loss_train = loss_tmp / (i+1)
        print(f"epoch {epoch}  loss_train {loss_train:.4f}")
    
    accelerator.save(model=model.state_dict(), save_directory="model.bin")
    
