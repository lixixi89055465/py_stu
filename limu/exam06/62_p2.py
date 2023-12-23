# -*- coding: utf-8 -*-
# @Time    : 2023/12/23 22:41
# @Author  : nanji
# @Site    : 
# @File    : 62_p2.py
# @Software: PyCharm 
# @Comment : 
import os
import torch
from torch import nn
from d2l import torch as d2l


class Seq2SeqEncoder(d2l.Encoder):
    def __init__(self, vocab_size, \
                 embed_size, \
                 num_hiddens, \
                 num_layer, \
                 dropout=0, \
                 **kwargs
                 ):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, \
                          num_layer, dropout=dropout)
    def forward(self,X,*args):
        X=self.embedding(X)
        X=X.permute(1,0,2)
        output,state=self.rnn(X)
        return output,state