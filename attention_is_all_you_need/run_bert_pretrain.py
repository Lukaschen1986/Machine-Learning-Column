# -*- coding: utf-8 -*-
"""
https://zh-v2.d2l.ai/chapter_natural-language-processing-pretraining/bert-pretraining.html
1-原始的BERT有两个版本，其中基本模型有1.1亿个参数，大模型有3.4亿个参数
2-在预训练BERT之后，我们可以用它来表示单个文本、文本对或其中的任何词元
3-在实验中，同一个词元在不同的上下文中具有不同的BERT表示。这支持BERT表示是上下文敏感的
"""
import os
# import math
import random
import torch as th
from torch import nn
from d2l import torch as d2l
from bert_model import BertModel

print(th.__version__)
print(th.version.cuda)
print(th.backends.cudnn.version())
th.set_default_tensor_type(th.DoubleTensor)
device = th.device("cuda" if th.cuda.is_available() else "cpu")

path = os.path.dirname(__file__)


# ---------------------------------------------------------------------------------------------------------------
d2l.DATA_HUB['wikitext-2'] = (
    'https://s3.amazonaws.com/research.metamind.io/wikitext/'
    'wikitext-2-v1.zip', '3c914d17d80b1459be871a5039ac23e752a53cbe')


def _read_wiki(data_dir):
    file_name = os.path.join(data_dir, 'wiki.train.tokens')
    with open(file_name, 'r', encoding="utf-8") as f:
        lines = f.readlines()
    # 大写字母转换为小写字母
    paragraphs = [line.strip().lower().split(' . ')
                  for line in lines if len(line.split(' . ')) >= 2]
    random.shuffle(paragraphs)
    return paragraphs


# ---------------------------------------------------------------------------------------------------------------
# 生成 下句预测模型 的数据
def _get_next_sentence(sentence, next_sentence, paragraphs):
    if random.random() < 0.5:
        is_next = True
    else:
        # paragraphs是三重列表的嵌套
        next_sentence = random.choice(random.choice(paragraphs))
        is_next = False
    return sentence, next_sentence, is_next


def _get_tokens_and_segments(tokens_a, tokens_b=None):
    tokens = ["<cls>"] + tokens_a + ["<sep>"]
    segments = [0] * (len(tokens_a) + 2)

    if tokens_b is not None:
        tokens += tokens_b + ["<sep>"]
        segments += [1] * (len(tokens_b) + 1)

    return tokens, segments


def _get_nsp_data_from_paragraph(paragraph, paragraphs, vocab, max_len):
    nsp_data_from_paragraph = []
    
    for i in range(len(paragraph)-1):
        tokens_a, tokens_b, is_next = _get_next_sentence(paragraph[i], paragraph[i+1], paragraphs)
        
        # 考虑1个'<cls>'词元和2个'<sep>'词元
        if len(tokens_a) + len(tokens_b) + 3 > max_len:
            continue
        
        tokens, segments = _get_tokens_and_segments(tokens_a, tokens_b)
        nsp_data_from_paragraph.append((tokens, segments, is_next))
    return nsp_data_from_paragraph


# ---------------------------------------------------------------------------------------------------------------
# 生成 掩码语言模型 的数据
def _replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds, vocab):
    # 为遮蔽语言模型的输入创建新的词元副本，其中输入可能包含替换的“<mask>”或随机词元
    mlm_input_tokens = [token for token in tokens]
    pred_positions_and_labels = []
    
    # 打乱后用于在遮蔽语言模型任务中获取15%的随机词元进行预测
    random.shuffle(candidate_pred_positions)
    
    for mlm_pred_position in candidate_pred_positions:
        if len(pred_positions_and_labels) >= num_mlm_preds:
            break
        masked_token = None
        
        # 80%的时间：将词替换为“<mask>”词元
        if random.random() < 0.8:
            masked_token = '<mask>'
        else:
            # 10%的时间：保持词不变
            if random.random() < 0.5:
                masked_token = tokens[mlm_pred_position]
            # 10%的时间：用随机词替换该词
            else:
                masked_token = random.choice(vocab.idx_to_token)
                
        mlm_input_tokens[mlm_pred_position] = masked_token
        pred_positions_and_labels.append((mlm_pred_position, tokens[mlm_pred_position]))
    return mlm_input_tokens, pred_positions_and_labels


def _get_mlm_data_from_tokens(tokens, vocab):
    candidate_pred_positions = []
    
    # tokens是一个字符串列表
    for i, token in enumerate(tokens):
        # 在遮蔽语言模型任务中不会预测特殊词元
        if token in ['<cls>', '<sep>']:
            continue
        candidate_pred_positions.append(i)
        
    # 遮蔽语言模型任务中预测15%的随机词元
    num_mlm_preds = max(1, round(len(tokens) * 0.15))
    mlm_input_tokens, pred_positions_and_labels = _replace_mlm_tokens(
        tokens, candidate_pred_positions, num_mlm_preds, vocab
        )
    pred_positions_and_labels = sorted(pred_positions_and_labels, key=lambda x: x[0])
    
    pred_positions = [v[0] for v in pred_positions_and_labels]
    mlm_pred_labels = [v[1] for v in pred_positions_and_labels]
    
    return vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labels]


# ---------------------------------------------------------------------------------------------------------------
# 将文本转换为预训练数据集
def _pad_bert_inputs(examples, max_len, vocab):
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, all_segments, valid_lens,  = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    
    for (token_ids, pred_positions, mlm_pred_label_ids, segments, is_next) in examples:
        all_token_ids.append(
            th.tensor(token_ids + [vocab['<pad>']] * (max_len - len(token_ids)), dtype=th.long)
            )
        all_segments.append(
            th.tensor(segments + [0] * (max_len - len(segments)), dtype=th.long)
            )
        
        # valid_lens不包括'<pad>'的计数
        valid_lens.append(th.tensor(len(token_ids), dtype=th.float32))
        all_pred_positions.append(
            th.tensor(pred_positions + [0] * (max_num_mlm_preds - len(pred_positions)), dtype=th.long)
            )
        
        # 填充词元的预测将通过乘以0权重在损失中过滤掉
        all_mlm_weights.append(
            th.tensor([1.0]*len(mlm_pred_label_ids) + [0.0]*(max_num_mlm_preds - len(pred_positions)), dtype=th.float32)
            )
        all_mlm_labels.append(
            th.tensor(mlm_pred_label_ids + [0]*(max_num_mlm_preds - len(mlm_pred_label_ids)), dtype=th.long)
            )
        nsp_labels.append(th.tensor(is_next, dtype=th.long))
        
    return (all_token_ids, all_segments, valid_lens, all_pred_positions, 
            all_mlm_weights, all_mlm_labels, nsp_labels)


class _WikiTextDataset(th.utils.data.Dataset):
    def __init__(self, paragraphs, max_len):
        # 输入paragraphs[i]是代表段落的句子字符串列表；
        # 而输出paragraphs[i]是代表段落的句子列表，其中每个句子都是词元列表
        paragraphs = [d2l.tokenize(paragraph, token='word') for paragraph in paragraphs]
        sentences = [sentence for paragraph in paragraphs for sentence in paragraph]
        self.vocab = d2l.Vocab(sentences, min_freq=5, reserved_tokens=['<pad>', '<mask>', '<cls>', '<sep>'])
        
        # 获取下一句子预测任务的数据
        examples = []
        for paragraph in paragraphs:
            examples.extend(_get_nsp_data_from_paragraph(paragraph, paragraphs, self.vocab, max_len))
            
        # 获取遮蔽语言模型任务的数据
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab) + (segments, is_next)) 
                    for tokens, segments, is_next in examples]
        
        # 填充输入
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights,
         self.all_mlm_labels, self.nsp_labels) = _pad_bert_inputs(examples, max_len, self.vocab)

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx], self.all_pred_positions[idx],
                self.all_mlm_weights[idx], self.all_mlm_labels[idx],
                self.nsp_labels[idx])

    def __len__(self):
        return len(self.all_token_ids)


def load_data_wiki(batch_size, max_len):
    """加载WikiText-2数据集"""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('wikitext-2', 'wikitext-2')
    paragraphs = _read_wiki(data_dir)
    '''
    paragraphs[0]
    [
     "<unk> route to the republican party 's presidential nomination , 
     nixon faced challenges from governor george romney of michigan , 
     governor nelson rockefeller of new york , governor ronald reagan of california , 
     and senator charles percy of illinois",
     
     'nixon won nine of the thirteen state primaries held that season , 
     although due to the population of his state , 
     governor reagan won the popular vote while carrying only california',
     
     'these victories , along with pledged delegate support from states not holding primaries , 
     secured nixon the nomination on the first ballot of the republican national convention , 
     where he named governor spiro agnew of maryland as his running mate .'
     ]
    '''
    train_set = _WikiTextDataset(paragraphs, max_len)
    train_iter = th.utils.data.DataLoader(train_set, batch_size, shuffle=True, num_workers=num_workers)
    return train_iter, train_set.vocab


# ---------------------------------------------------------------------------------------------------------------
# 模型训练
def _get_batch_loss_bert(net, objt, vocab_size, tokens_X, segments_X, valid_lens_x, 
                         pred_positions_X, mlm_weights_X, mlm_Y, nsp_y):
    # 前向传播
    _, mlm_Y_hat, nsp_Y_hat = net(tokens_X, segments_X, valid_lens_x.reshape(-1), pred_positions_X)
    
    # 计算遮蔽语言模型损失
    mlm_l = objt(mlm_Y_hat.reshape(-1, vocab_size), mlm_Y.reshape(-1)) * mlm_weights_X.reshape(-1, 1)
    mlm_l = mlm_l.sum() / (mlm_weights_X.sum() + 1e-8)
    
    # 计算下一句子预测任务的损失
    nsp_l = objt(nsp_Y_hat, nsp_y)
    l = mlm_l + nsp_l
    return mlm_l, nsp_l, l


def train_bert(train_iter, net, objt, vocab_size, devices, num_steps):
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    trainer = th.optim.Adam(net.parameters(), lr=0.01)
    step, timer = 0, d2l.Timer()
    animator = d2l.Animator(xlabel='step', ylabel='loss', xlim=[1, num_steps], legend=['mlm', 'nsp'])
    
    # 遮蔽语言模型损失的和，下一句预测任务损失的和，句子对的数量，计数
    metric = d2l.Accumulator(4)
    num_steps_reached = False
    
    while step < num_steps and not num_steps_reached:
        for tokens_X, segments_X, valid_lens_x, pred_positions_X, mlm_weights_X, mlm_Y, nsp_y in train_iter:
            tokens_X = tokens_X.to(devices[0])
            segments_X = segments_X.to(devices[0])
            valid_lens_x = valid_lens_x.to(devices[0])
            pred_positions_X = pred_positions_X.to(devices[0])
            mlm_weights_X = mlm_weights_X.to(devices[0])
            mlm_Y, nsp_y = mlm_Y.to(devices[0]), nsp_y.to(devices[0])
            
            trainer.zero_grad()
            timer.start()
            mlm_l, nsp_l, l = _get_batch_loss_bert(
                net, objt, vocab_size, tokens_X, segments_X, valid_lens_x,
                pred_positions_X, mlm_weights_X, mlm_Y, nsp_y
                )
            l.backward()
            trainer.step()
            
            metric.add(mlm_l, nsp_l, tokens_X.shape[0], 1)
            timer.stop()
            animator.add(step + 1,
                         (metric[0] / metric[3], metric[1] / metric[3]))
            
            step += 1
            if step == num_steps:
                num_steps_reached = True
                break

    print(f'MLM loss {metric[0] / metric[3]:.3f}, '
          f'NSP loss {metric[1] / metric[3]:.3f}')
    print(f'{metric[2] / timer.sum():.1f} sentence pairs/sec on '
          f'{str(devices)}')


# ---------------------------------------------------------------------------------------------------------------
# 用BERT表示文本
def get_bert_encoding(net, tokens_a, tokens_b=None):
    tokens, segments = _get_tokens_and_segments(tokens_a, tokens_b)
    token_ids = th.tensor(vocab[tokens], device=devices[0]).unsqueeze(0)
    segments = th.tensor(segments, device=devices[0]).unsqueeze(0)
    valid_len = th.tensor(len(tokens), device=devices[0]).unsqueeze(0)
    encoded_X, _, _ = net(token_ids, segments, valid_len, pred_positions=None)
    return encoded_X



if __name__ == "__main__":
    # 加载数据
    batch_size, max_len = 512, 64
    train_iter, vocab = load_data_wiki(batch_size, max_len)
    
    # 预训练
    net = BertModel(len(vocab), num_hiddens=128, norm_shape=[128],
                    ffn_num_input=128, ffn_num_hiddens=256, num_heads=2,
                    num_layers=2, dropout=0.2, key_size=128, query_size=128,
                    value_size=128, hid_in_features=128, mlm_in_features=128,
                    nsp_in_features=128)
    devices = d2l.try_all_gpus()
    objt = nn.CrossEntropyLoss()
    train_bert(train_iter, net, objt, len(vocab), devices, 50)
    
    # 抽取特征-一个句子
    tokens_a = ['a', 'crane', 'is', 'flying']
    encoded_text = get_bert_encoding(net, tokens_a)
    # 词元：'<cls>','a','crane','is','flying','<sep>'
    encoded_text_cls = encoded_text[:, 0, :]
    encoded_text_crane = encoded_text[:, 2, :]
    encoded_text.shape, encoded_text_cls.shape, encoded_text_crane[0][:3]
    
    # 抽取特征-一对句子
    tokens_a, tokens_b = ['a', 'crane', 'driver', 'came'], ['he', 'just', 'left']
    encoded_pair = get_bert_encoding(net, tokens_a, tokens_b)
    # 词元：'<cls>','a','crane','driver','came','<sep>','he','just',
    # 'left','<sep>'
    encoded_pair_cls = encoded_pair[:, 0, :]
    encoded_pair_crane = encoded_pair[:, 2, :]
    encoded_pair.shape, encoded_pair_cls.shape, encoded_pair_crane[0][:3]
    

