# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings("ignore")
import os
import transformers
import numpy as np
import pandas as pd
import torch as th
import torch.optim as optim
import torch.nn.functional as F

from dataclasses import dataclass
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig)
from peft import (LoraConfig, get_peft_model) 


# ----------------------------------------------------------------------------------------------------------------
# 基础设置
th.autograd.set_detect_anomaly(True)
device = th.device("cuda" if th.cuda.is_available() else "cpu")
device_cnt = th.cuda.device_count()
print(f"device = {device}; device_cnt = {device_cnt}")
print(f"torch version = {th.__version__}")
print(f"cuda version = {th.version.cuda}")
print(f"transformers version = {transformers.__version__}")
'''
This is my PC:
device = cuda; device_cnt = 1
torch version = 2.10.0+cu126
cuda version = 12.6
transformers version = 5.2.0
'''

# ----------------------------------------------------------------------------------------------------------------
# 定义路径
path_project = "C:/my_project/MyGit/Machine-Learning-Column/explore_on_llm"
path_data = os.path.join(os.path.dirname(path_project), "data")
path_model = "F:/LLM"
path_output = os.path.join(path_model, "output")

# ----------------------------------------------------------------------------------------------------------------
# 加载数据集
dataset = load_dataset(
    path="csv",
    data_files=os.path.join(path_data, "chengyu.csv"),
    split="all"
    )
print(dataset[1])

# ----------------------------------------------------------------------------------------------------------------
# 定义模型参数
@dataclass
class ModelArgs:
    max_seq_len: int = 512
    # 此处省略其他参数，后续根据需要添加

# ----------------------------------------------------------------------------------------------------------------
# 定义量化配置
config_bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=th.bfloat16,
    bnb_4bit_use_double_quant=True
)  # QLoRA

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,  # lora_alpha是一个缩放因子，通常设置为r的倍数，以确保lora权重的初始范围适当。可以帮助模型更快地收敛。
    lora_dropout=0.1,  # lora_dropout是应用于lora权重的dropout概率，可以帮助防止过拟合。通常设置为0.1或0.2。
    use_rslora=True,  # 是否使用RSLora，RSLora是一种改进的LoRA方法，可以在某些情况下提高性能。根据需要选择是否使用。
    bias="none",  # 是否微调偏置项，通常设置为"none"，表示不微调偏置项，以减少训练参数量和计算复杂度。
    task_type="CAUSAL_LM",  # 任务类型，CAUSAL_LM表示因果语言建模，适用于生成任务。根据实际任务类型选择合适的值。
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "lm_head"
        ]
    )

# ----------------------------------------------------------------------------------------------------------------
# 生成器：根据用户的提问，给出回答；根据反思器的修正建议，优化回答
class Generator(th.nn.Module):
    """
    checkpoint = generator_checkpoint
    self = Generator(path_model, checkpoint)
    """
    def __init__(self, path_model, checkpoint):
        super(Generator, self).__init__()
        self.path_model = path_model
        self.checkpoint = checkpoint
        self._kwargs = {"max_new_tokens": 512, "temperature": 0.2, "top_p": 0.1, "do_sample": True}
        
        self._tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=os.path.join(path_model, checkpoint),
            cache_dir=path_model,
            force_download=False,
            local_files_only=True,
            )
        self._tokenizer.pad_token = (self._tokenizer.eos_token if self._tokenizer.pad_token is None else self._tokenizer.pad_token)
        self._tokenizer.padding_size = "left"
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=os.path.join(path_model, checkpoint),
            cache_dir=path_model,
            force_download=False,
            local_files_only=True,
            device_map="auto",
            low_cpu_mem_usage=True,
            dtype=th.bfloat16,
            # quantization_config=config_bnb,
            )
        self._model = get_peft_model(model=self.base_model, peft_config=lora_config)
        print(f"Generator:\n{self._model}")
        
        self._model.gradient_checkpointing_enable()  # 启用梯度检查点，节省显存
        self._model.enable_input_require_grads()  # 使能输入的梯度计算，以便后续计算生成器的损失
        self._model.config.use_cache = False  # 禁用KV缓存，以便每次前向传播都能得到完整的隐藏状态
        
        self._system_prompt = "你是一个博学多才的汉语言文学家，精通成语和俗语的含义和出处。"
        
    def get_model_inputs(self, user_prompt):
        """
        user_prompt = "解释以下：蜻蜓点水"
        len(text)
        """
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_prompt},
            ]
        
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
            )
        
        model_inputs = self._tokenizer(
            text=text,
            max_length=300,  # 根据实际情况调整输入最大长度
            padding="max_length",
            return_tensors="pt",
            ).to(device)
        return model_inputs
        
    def forward(self, user_prompt):
        model_inputs = self.get_model_inputs(user_prompt)
        
        outputs = self._model(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            **self._kwargs,
            )
        generator_logits = outputs.logits  # 形状为[batch_size, seq_len, vocab_size]
        return generator_logits
    
    def generate(self, user_prompt):
        model_inputs = self.get_model_inputs(user_prompt)
        
        self._model.eval()
        with th.inference_mode():
            complete_ids = self._model.generate(
                input_ids=model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                **self._kwargs,
                )
        
        input_ids = model_inputs.input_ids
        generated_ids = [O[len(I): ] for (I, O) in zip(input_ids, complete_ids)]
        generator_response = self._tokenizer.batch_decode(
            sequences=generated_ids, 
            skip_special_tokens=True
            )[0]
        return generator_response

    def revise(self, user_prompt, generator_response, suggestion):
        prompt = (
            "根据提问、初始回答、修正建议，输出优化后的回答。"
            "注意事项：直接给出优化后的回答，不要输出思考过程等多余的文字。\n"
            "思考步骤：\n"
            "1、理解提问和初始回答\n"
            "2、认真领会修正建议，明确优化方向\n"
            "3、组织语言，输出优化后的回答\n"
            f"提问：{user_prompt}\n"
            f"初始回答：{generator_response}\n"
            f"修正建议：{suggestion}\n"
            "优化后的回答："
            )
        return self.forward(prompt), self.generate(prompt)


# ----------------------------------------------------------------------------------------------------------------
# 反思器：分析生成器的回答，给出修正建议
class Reflector(th.nn.Module):
    """
    checkpoint = "Qwen/Qwen2.5-4B-Instruct"
    self = Reflector(path_model, checkpoint)
    """
    def __init__(self, path_model, checkpoint):
        super(Reflector, self).__init__()
        self.path_model = path_model
        self.checkpoint = checkpoint
        self._kwargs = {"max_new_tokens": 512, "temperature": 0.2, "top_p": 0.1, "do_sample": True}
        
        self._tokenizer =AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=os.path.join(path_model, checkpoint),
            cache_dir=path_model,
            force_download=False,
            local_files_only=True,
            )
        self._tokenizer.pad_token = (self._tokenizer.eos_token if self._tokenizer.pad_token is None else self._tokenizer.pad_token)
        self._tokenizer.padding_size = "left"
        
        self._model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=os.path.join(path_model, checkpoint),
            cache_dir=path_model,
            force_download=False,
            local_files_only=True,
            device_map="auto",
            low_cpu_mem_usage=True,
            dtype=th.bfloat16,
            quantization_config=config_bnb,
            )
        for param in self._model.parameters():
            param.requires_grad_(False)
        print(f"Reflector:\n{self._model}")
        
        self._system_prompt = "你是一个博学多才的汉语言文学家，擅长对他人的回答进行分析，给出修正建议。"
        
        self.score_head = th.nn.Sequential(
            # layer-1
            th.nn.Linear(in_features=3584, out_features=256, bias=False, dtype=th.bfloat16),
            th.nn.GELU(),
            th.nn.Dropout(p=0.2),
            th.nn.LayerNorm(normalized_shape=256, dtype=th.bfloat16),
            # layer-2
            th.nn.Linear(in_features=256, out_features=128, bias=False, dtype=th.bfloat16),
            th.nn.GELU(),
            th.nn.Dropout(p=0.2),
            th.nn.LayerNorm(normalized_shape=128, dtype=th.bfloat16),
            # layer-3
            th.nn.Linear(in_features=128, out_features=64, bias=False, dtype=th.bfloat16),
            th.nn.GELU(),
            th.nn.Dropout(p=0.2),
            th.nn.LayerNorm(normalized_shape=64, dtype=th.bfloat16),
            # layer-out
            th.nn.Linear(in_features=64, out_features=1, bias=False, dtype=th.bfloat16),
            ).to(device)
        
    def get_model_inputs(self, user_prompt, assistant_prompt):
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_prompt},
            ]
        
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
            )
        
        model_inputs = self._tokenizer(
            text=text,
            max_length=512,
            padding=True,
            return_tensors="pt",
            ).to(device)
        return model_inputs
    
    def get_model_inputs_for_reflect(self, prompt):
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": prompt},
            ]
        
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
            )
        
        model_inputs = self._tokenizer(
            text=text,
            max_length=512,
            padding=True,
            return_tensors="pt",
            ).to(device)
        return model_inputs
    
    def forward(self, user_prompt, assistant_prompt):
        model_inputs = self.get_model_inputs(user_prompt, assistant_prompt)

        outputs = self._model(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            output_hidden_states=True,  # 需要输出隐藏状态，以便后续计算得分
            **self._kwargs,
            )
        
        last_hidden_state = outputs.hidden_states[-1][:, -1, :]  # 取最后一个token的隐藏状态作为输入，形状为[batch_size, hidden_size]
        reflector_logits = self.score_head(last_hidden_state)  # 通过得分头得到反思器的输出，形状为[batch_size, 1]
        return reflector_logits
    
    def reflect(self, user_prompt, standard_response, generator_response):
        prompt = f"""根据问题和参考答案，分析回答的不足之处，并给出具体的修正建议。
        问题：\n{user_prompt}\n
        参考答案：\n{standard_response}\n
        回答：\n{generator_response}\n
        修正建议：\n
        """
        model_inputs = self.get_model_inputs_for_reflect(prompt)
        
        self._model.eval()
        with th.inference_mode():
            complete_ids = self._model.generate(
                input_ids=model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                **self._kwargs,
                )
        
        input_ids = model_inputs.input_ids
        generated_ids = [O[len(I): ] for (I, O) in zip(input_ids, complete_ids)]
        suggestion = self._tokenizer.batch_decode(
            sequences=generated_ids, 
            skip_special_tokens=True
            )[0]
        return suggestion

# ----------------------------------------------------------------------------------------------------------------
# 定义损失函数
class LossFunction(object):
    def __init__(self, args):
        self.args = args
        
    def reflector_loss(self, logits, labels):
        criterion = th.nn.BCEWithLogitsLoss()
        reflector_loss = criterion(logits, labels)
        return reflector_loss
    
    def revise_loss(self, generator_logits, standard_ids):
        criterion = th.nn.CrossEntropyLoss(ignore_index=self.args.pad_token_id)
        shift_logits = generator_logits[:, 0:-1, :].flatten(start_dim=0, end_dim=1)
        shift_labels = standard_ids[:, 1: ].flatten(start_dim=0, end_dim=1)
        revise_loss = criterion(shift_logits, shift_labels)
        return revise_loss
    
    def kl_loss(self, revise_logits, generator_logits, reduction="batchmean"):
        revise_logits_detached = revise_logits.detach()
        kl_loss = F.kl_div(
            input=F.log_softmax(generator_logits, dim=-1),
            target=F.softmax(revise_logits_detached, dim=-1),
            reduction=reduction,
            )
        return kl_loss
    
# ----------------------------------------------------------------------------------------------------------------
# 统计模型的可训练参数量和总参数量
def print_trainable_parameters(model):
    trainable_params = 0
    all_params = 0

    for param in model.parameters():
        if param.requires_grad:
            trainable_params += param.numel()
        all_params += param.numel()
    
    print(f"trainable params: {trainable_params} || all params: {all_params} || trainable%: {100 * trainable_params / all_params:.2f}%")



if __name__ == "__main__":
    # 初始化生成器和反思器
    generator_checkpoint = "Qwen/Qwen3.5-4B"
    reflector_checkpoint = "Qwen/Qwen3.5-4B"
    
    xxx
    G = Generator(path_model, generator_checkpoint)
    R = Reflector(path_model, reflector_checkpoint)
    
    print("Generator trainable parameters:")
    print_trainable_parameters(G)
    
    print("Reflector trainable parameters:")
    print_trainable_parameters(R)
    
    # 构建数据加载器和损失函数
    dataloader = th.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    loss_fn = LossFunction(args=ModelArgs())
    
    # 构建优化器
    g_optim = optim.AdamW(params=G.parameters(), lr=0.0001, beta=(0.9, 0.999), eps=10**-8, weight_decay=0.01)
    r_optim = optim.AdamW(params=R.parameters(), lr=0.0001, beta=(0.9, 0.999), eps=10**-8, weight_decay=0.01)
    
    # 训练循环
    epoch = 1
    reflector_iters = 2
    generator_total_loss = 0.0
    reflector_total_loss = 0.0
    
    for e in range(epoch):
        for batch in dataloader:
            # 准备输入数据
            user_prompt = batch["user_prompt"][0]
            standard_response = batch["standard_response"][0]
            
            standard_ids = G._tokenizer(
                text=standard_response,
                max_length=256,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)
            
            # Generator生成回答，并计算反思器的损失，更新反思器参数
            generator_logits = G.forward(user_prompt)
            generator_response = G.generate(user_prompt)
            reflector_cum_loss = 0.0
            
            for _ in range(reflector_iters):
                # 计算反思器的损失
                reflector_logits_1 = R.forward(user_prompt, standard_response)
                reflector_logits_0 = R.forward(user_prompt, generator_response)
                
                # 反思器的目标是给出积极的修正建议，因此标准回答的标签为1，生成回答的标签为0
                reflector_loss_1 = loss_fn.reflector_loss(reflector_logits_1, th.ones_like(reflector_logits_1))
                reflector_loss_0 = loss_fn.reflector_loss(reflector_logits_0, th.ones_like(reflector_logits_0))
                reflector_loss = reflector_loss_1 + reflector_loss_0
                
                # 更新反思器参数
                r_optim.zero_grad()
                reflector_loss.backward()
                r_optim.step()
            
            # 计算反思器的平均损失
            reflector_cum_loss /= reflector_iters
            
            # 反思器给出修正建议
            suggestion = R.reflect(user_prompt, standard_response, generator_response)
            
            # Generator根据修正建议优化回答，并计算生成器的损失，更新生成器参数
            revise_logits, revised_response = G.revise(user_prompt, generator_response, suggestion)
            
            # 计算生成器的损失：包括修正损失、反思器的对抗损失、以及KL散度损失
            gen_loss = loss_fn.revise_loss(revise_logits, standard_ids)
            reflector_logits_0 = R.forward(user_prompt, generator_response)
            reflector_loss_adv = loss_fn.reflector_loss(reflector_logits_0, th.ones_like(reflector_logits_0))
            kl_loss = loss_fn.kl_loss(revise_logits, generator_logits)
            generator_loss = gen_loss + reflector_loss_adv + 0.001 * kl_loss
            
            # 更新生成器参数
            g_optim.zero_grad()
            generator_loss.backward()
            g_optim.step()
            
            # 统计损失
            generator_total_loss += generator_loss.item()
            reflector_total_loss += reflector_cum_loss
            print(f"reflector_loss: {reflector_cum_loss:.4f} | generator_loss: {generator_loss.item():.4f}")
        
        # 计算并打印平均损失
        generator_avg_loss = generator_total_loss / len(dataloader)
        reflector_avg_loss = reflector_total_loss / len(dataloader)
        print(f"Epoch {e + 1}/{epoch} | Average Reflector Loss: {reflector_avg_loss:.4f} | Average Generator Loss: {generator_avg_loss:.4f}")
        
            
            
         




