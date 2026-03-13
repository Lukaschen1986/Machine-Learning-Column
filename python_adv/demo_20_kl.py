# -*- coding: utf-8 -*-
import torch as th
import torch.nn.functional as F


def calculate_kl_between_two_logits(
    logits1: th.Tensor,  # p
    logits2: th.Tensor,  # q
    reduction: str = "mean"
) -> float:
    """
    计算两个logits对应的生成分布之间的KL散度
    :param logits1: 第一个logits，维度支持 [seq_len, vocab_size] 或 [batch_size, seq_len, vocab_size]
    :param logits2: 第二个logits，维度需与logits1完全一致
    :param kl_direction: KL散度方向，"1||2"表示KL(p1||p2)（让p2对齐p1），"2||1"表示KL(p2||p1)
    :param reduction: 结果汇总方式，"mean"（平均）/"sum"（求和）/"none"（返回逐token结果）
    :return: KL散度值（若reduction为"none"则返回张量，否则返回标量）
    """
    # 1. 校验维度一致性
    if logits1.shape != logits2.shape:
        raise ValueError(f"logits1和logits2维度不一致！logits1: {logits1.shape}, logits2: {logits2.shape}")
    
    # 2. 将logits转为概率分布（softmax，对最后一维（vocab_size）计算）
    prob1 = F.softmax(logits1, dim=-1)  # 与logits1维度一致
    prob2 = F.softmax(logits2, dim=-1)  # 与logits2维度一致
    
    # 3. 添加极小值epsilon，避免log(0)产生无穷大（工程必备）
    epsilon = 1e-8
    prob1 = prob1 + epsilon
    prob2 = prob2 + epsilon
    
    # 4. 选择KL散度方向并计算
    # KL(p1||p2) = sum(p1 * log(p1/p2))，即让p2对齐p1的差异
    kl_per_token = th.sum(prob1 * th.log(prob1 / prob2), dim=-1)

    # 5. 结果汇总（适配不同维度场景）
    if reduction == "none":
        return kl_per_token  # 返回逐token/逐样本的KL散度张量
    elif reduction == "sum":
        return kl_per_token.sum().item()  # 所有位置的KL散度求和
    elif reduction == "mean":
        return kl_per_token.mean().item()  # 所有位置的KL散度平均
    else:
        raise ValueError("reduction仅支持'none'/'sum'/'mean'")


def calculate_kl_divergence(logits_p, logits_q, reduction='batchmean'):
    """
    计算两个logits分布之间的KL散度
    
    参数:
        logits_p: 第一个模型的输出，形状通常为 [batch_size, vocab_size] 或 [batch_size, seq_len, vocab_size]
        logits_q: 第二个模型的输出，形状需与logits_p完全一致
        reduction: 损失聚合方式，'batchmean'表示对批次求平均（符合KL散度数学定义），
                   'mean'会额外除以维度数，'sum'是求和，'none'返回逐元素结果
    
    返回:
        kl_div: 计算得到的KL散度值
    """
    # 1. 将logits转换为对数概率（log_softmax = log(softmax(logits))）
    log_probs_q = F.log_softmax(logits_q, dim=-1)  # dim=-1表示对最后一维（词表维度）做归一化
    
    # 2. 将logits转换为普通概率（softmax）
    probs_p = F.softmax(logits_p, dim=-1)
    
    # 3. 计算KL散度：KL(log_probs_p || probs_q)
    # 注意：F.kl_div的第一个参数是log_prob，第二个是prob，reduction='batchmean'是推荐的正确方式
    kl_div = F.kl_div(
        input=log_probs_q, 
        target=probs_p, 
        reduction=reduction
        )
    return kl_div



# ------------------- 测试示例 -------------------
if __name__ == "__main__":
    # 模拟大模型输出的logits：批次大小=2，序列长度=3，词表大小=10
    batch_size, seq_len, vocab_size = 2, 3, 10
    logits1 = th.randn(batch_size, seq_len, vocab_size)  # 模型1输出
    logits2 = th.randn(batch_size, seq_len, vocab_size)  # 模型2输出
    
    calculate_kl_between_two_logits(logits1, logits2, reduction="mean")
    calculate_kl_divergence(logits1, logits2, reduction="batchmean")
    
    
