# -*- coding: utf-8 -*-
# @Time    : 2023/12/17 22:52
# @Author  : nanji
# @Site    : 
# @File    : 53_2.py
# @Software: PyCharm 
# @Comment :
import random
import torch
from d2l import torch as d2l

tokens = d2l.tokenize(d2l.read_time_machine())
corpus = [token for line in tokens for token in line]
vocab=d2l.Vocab(corpus)
print(vocab.token_freqs[:10])
