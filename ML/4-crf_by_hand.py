# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings("ignore")
import torch as tc
import torch.nn as nn
import torch.optim as optim


print(tc.__version__)
print(tc.version.cuda)
print(tc.backends.cudnn.version())
tc.manual_seed(1)
tc.set_default_tensor_type(tc.DoubleTensor)


# 数据预处理
def prepare_sequence(lst, dct):
    # 将原始序列转化为索引序列
    lst_idxs = [dct.get(x) for x in lst]
    tnr_idxs = tc.Tensor(lst_idxs, dtype=tc.long)
    return tnr_idxs


# 定义网络结构
class BiLSTM_CRF(nn.Module):
    def __init__(self, config):
        super(BiLSTM_CRF, self).__init__()
        self.vocab_size = config.get("vocab_size")
        self.tag_to_idx = config.get("tag_to_idx")
        self.tagset_size = len(self.tag_to_idx)
        self.embedding_dim = config.get("embedding_dim")
        self.hidden_dim = config.get("hidden_dim")
        self.START_TAG = config.get("START_TAG")
        self.STOP_TAG = config.get("STOP_TAG")
        
        # init layers
        self.embedding_layer = nn.Embedding(num_embeddings=self.vocab_size,
                                            embedding_dim=self.embedding_dim)
        
        self.lstm_layer = nn.LSTM(input_size=self.embedding_dim,
                                  hidden_size=self.hidden_dim,
                                  num_layers=1,
                                  dropout=config.get("dropout"),
                                  batch_first=False,  # Default
                                  bidirectional=True)
        
        self.fc_layer = nn.Linear(in_features=self.hidden_dim*2,
                                  out_features=self.tagset_size)
        
        self.transition_layer = nn.Parameter(tc.randn([self.tagset_size, self.tagset_size]))
        self.transition_layer.data[self.tag_to_idx.get(self.START_TAG), :] = -10000.0
        self.transition_layer.data[:, self.tag_to_idx.get(self.STOP_TAG)] = -10000.0
        
        
    @staticmethod
    def argmax(tnr_scores):
        # 按行取最大值对应的索引
        tnr_val, tnr_idx = tc.max(tnr_scores, dim=1)
        idx = tnr_idx.item()
        return idx
        
    
    @staticmethod
    def log_sum_exp(tnr_scores):
        """
        tc.log(tc.sum(tc.exp(tnr_scores)))
        """
        max_score = tc.max(tnr_scores)
        max_score_broadcast = max_score.reshape(1, -1).expand(1, tnr_scores.shape[1])
        alpha = max_score + tc.log(tc.sum(tc.exp(tnr_scores - max_score_broadcast)))
        return alpha
    
    
    def f(self, tnr_word_idxs, tnr_tag_idxs):
        lstm_feats = self._get_lstm_features(tnr_word_idxs)
        terminal_total_score = self._get_total_score(lstm_feats)
        terminal_real_score = self._get_real_score(lstm_feats, tnr_tag_idxs)
        return terminal_real_score, terminal_total_score
    
    
    def _get_lstm_features(self, tnr_word_idxs):
        embeds = self.embedding_layer(tnr_word_idxs)
        embeds = embeds.reshape(len(tnr_word_idxs), 1, -1)
        
        h0 = tc.randn([2, 1, self.hidden_dim]) * 0.01
        c0 = tc.randn([2, 1, self.hidden_dim]) * 0.01
        lstm_out, [ht, ct] = self.lstm_layer(embeds, [h0, c0])
        lstm_out = lstm_out.resahpe(len(tnr_word_idxs), self.hidden_dim)
        
        lstm_feats = self.fc_layer(lstm_out)
        return lstm_feats
    
    
    def _get_total_score(self, lstm_feats):
        # 计算全部路径的前向得分，对应损失函数的第三项
        tnr_init_scores = tc.full((1, self.tagset_size), -10000.0)
        tnr_init_scores[0][self.tag_to_ix[self.START_TAG]] = 0.0
        previous_socres = tnr_init_scores

        # Iterate through the sentence
        for feat in lstm_feats:
            lst_socres = []
            
            for next_tag in range(self.tagset_size):
                emission_scores = feat[next_tag].reshape(1, -1).expand(1, self.tagset_size)
                transition_scores = self.transition_layer[next_tag].reshape(1, -1)
                tmp_scores = previous_socres + emission_scores + transition_scores
                score = BiLSTM_CRF.log_sum_exp(tmp_scores)
                score = score.reshape(1)
                lst_socres.append(score)
            
            # 更新 previous_socres
            previous_socres = tc.cat(lst_socres, dim=0)
            previous_socres = previous_socres.reshape(1, -1)
        
        # 计算最终的前向得分
        total_scores = previous_socres + self.transition_layer[self.tag_to_idx.get(self.STOP_TAG)]
        terminal_total_score = BiLSTM_CRF.log_sum_exp(total_scores)
        return terminal_total_score
    
    
    def _get_real_score(self, lstm_feats, tnr_tag_idxs):
        # 计算真实路径的前向得分，对应损失函数的前两项
        score = tc.zeros(1)
        tnr_tag_idxs = tc.cat([tc.tensor([self.tag_to_ix[self.START_TAG]], dtype=tc.long), tnr_tag_idxs])
        
        for (i, feat) in enumerate(lstm_feats):
            score = score + \
                    feat[tnr_tag_idxs[i+1]] + \
                    self.transition_layer[tnr_tag_idxs[i+1], tnr_tag_idxs[i]]
        
        terminal_real_score = score + \
                              self.transition_layer[self.tag_to_idx[self.STOP_TAG], tnr_tag_idxs[-1]]
        return terminal_real_score
    
    
    def decode(self, tnr_word_idxs):
        lstm_feats = self._get_lstm_features(tnr_word_idxs)
        score, tag_seq = self._viterbi(lstm_feats)
        return score, tag_seq
    
    
    def _viterbi(self, lstm_feats):
        lst_backpointers = []
        tnr_init_scores = tc.full((1, self.tagset_size), -10000.0)
        tnr_init_scores[0][self.tag_to_ix[self.START_TAG]] = 0.0
        previous_socres = tnr_init_scores
        
        for feat in lstm_feats:
            lst_bptrs = []
            lst_viterbivars = []

            for next_tag in range(self.tagset_size):
                transition_scores = self.transition_layer[next_tag].reshape(1, -1)
                tmp_scores = previous_socres + transition_scores
                best_tag_id = BiLSTM_CRF.argmax(tmp_scores)
                lst_bptrs.append(best_tag_id)
                lst_viterbivars.append(tmp_scores[0][best_tag_id].reshape(1))
            
            previous_socres = tc.cat(lst_viterbivars, dim=0) + feat
            previous_socres = previous_socres.reshape(1, -1)
            lst_backpointers.append(lst_bptrs)

        # Transition to STOP_TAG
        total_scores = previous_socres + self.transition_layer[self.tag_to_idx.get(self.STOP_TAG)]
        best_tag_id = BiLSTM_CRF.argmax(total_scores)
        path_score = total_scores[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for lst_bptrs in reversed(lst_backpointers):
            best_tag_id = lst_bptrs[best_tag_id]
            best_path.append(best_tag_id)
            
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[self.START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path
    
    

if __name__ == "__main__":
    # 构造数据集
    lst_bio = [
            (["a", "p", "p"], ["B", "I", "I"]),
            ]
    
    # word_to_idx
    set_words = set()
    for (lst_words, lst_tags) in lst_bio:
        tmp = set(lst_words)
        set_words = set_words.union(tmp)
    
    word_to_idx = {word: idx for (idx, word) in enumerate(set_words)}
    vocab_size = len(word_to_idx)
    
    # tag_to_idx
    tag_to_idx = {"B": 0, "I": 1, "O": 2, "<start>": 3, "<stop>": 4}
    
    # 模型训练
    config = {...}
    model = BiLSTM_CRF(config)
    opti = optim.Adam(params=model.parameters(), lr=0.1, betas=(0.9, 0.999), eps=10**-8, weight_decay=0.01)
#    opti = optim.SGD(params=model.parameters(), lr=0.01, momentum=0.9)
    objt = lambda real_score, total_score: -(real_score - total_score)
    
    for epoch in range(100):
        loss_tmp = 0
        model.train()
        for (lst_words, lst_tags) in lst_bio:
            tnr_word_idxs = prepare_sequence(lst_words, word_to_idx)
            tnr_tag_idxs = prepare_sequence(lst_tags, tag_to_idx)
            
            real_score, total_score = model.f(tnr_word_idxs, tnr_tag_idxs)
            loss = objt(real_score, total_score)
            loss_tmp += loss.item()
            
            opti.zero_grad()
            loss.backward()
            opti.step()
        
        loss_tmp = loss_tmp / len(lst_bio)
        print(f"epoch {epoch} loss_train {loss_tmp:.4f}")
        
        model.eval()
        with tc.no_grad():
            tnr_word_idxs = prepare_sequence(lst_bio[0][0], word_to_idx)
            print(model.decode(tnr_word_idxs))
        
        
        
