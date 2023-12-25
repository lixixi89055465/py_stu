# -*- coding: utf-8 -*-
# @Time : 2023/12/25 21:27
# @Author : nanji
# @Site : 
# @File : 65_p2.py
# @Software: PyCharm 
# @Comment :
import os
import torch
from torch import nn
from d2l import torch as d2l
import math


def masked_softmax(X, valid_lens):
    '''通过在最后一个轴上遮蔽元素来执行softmax 操作 '''
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        X = X.reshape(-1, shape[-1])
        maxlen = X.size(1)
        mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] \
               < valid_lens[:, None]
        X[~mask] = -1e6
        # X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
        #                       value=-1e6)
        # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
        return nn.functional.softmax(X.reshape(shape), dim=-1)


# masked_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3]))
# masked_softmax(torch.rand(2, 2, 4), torch.tensor([[1, 3], [2, 4]]))
class AdditiveAttention(nn.Module):
    '''加性注意力 '''

    def __init__(self, key_size, query_size, num_hiddens, \
                 dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.W_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = dropout

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(self.keys)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        scores = self.W_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


queries, keys = torch.normal(0, 1, (2, 1, 10)), torch.ones((2, 10, 2))
values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4). \
    repeat(2, 1, 1)

attention = AdditiveAttention(key_size=2, query_size=20, \
                              num_hiddens=10, dropout=0.1)
attention.eval()
attention(queries, keys, values, valid_lens)
