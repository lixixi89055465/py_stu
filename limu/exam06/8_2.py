# -*- coding: utf-8 -*-
# @Time : 2023/12/28 13:15
# @Author : nanji
# @Site : 
# @File : 8_2.py
# @Software: PyCharm 
# @Comment : 
import os
import torch
from torch import nn
from d2l import torch as d2l
import re
import collections

# @save
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
								'090b5e7e70c295757f55df93cb0a180b9691891a')


def read_time_machine():  # @save
	"""将时间机器数据集加载到文本行的列表中"""
	with open(d2l.download('time_machine'), 'r') as f:
		lines = f.readlines()
	return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]


lines = read_time_machine()
print(f'# 文本总行数: {len(lines)}')
print(lines[0])
print('0' * 100)
print(lines[10])


def tokenize(lines, token='word'):
	if token == 'word':
		return [line.split() for line in lines]
	elif token == 'char':
		return [list(line) for line in lines]
	else:
		print('错误：未知词元类型：' + token)


# tokens = tokenize(lines)


# print('2' * 100)
# for i in range(11):
# 	print(tokens[i])
class Vocab:
	def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
		if tokens is None:
			tokens = []
		if reserved_tokens is None:
			reserved_tokens = []
		# 按出现的频率排序
		counter = count_corpus(tokens)
		self._token_freqs = sorted(counter.items(), key=lambda x: x[1], \
								   reverse=True)
		# 未知词元的索引为0
		self.idx_to_token = ['<unk>'] + reserved_tokens
		self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
		for token, freq in self._token_freqs:
			if freq < min_freq:
				break
			if token not in self.token_to_idx:
				self.idx_to_token.append(token)
				self.token_to_idx[token] = len(self.idx_to_token) - 1

	def __len__(self):
		return len(self.idx_to_token)

	def __getitem__(self, tokens):
		if not isinstance(tokens, (list, tuple)):
			return self.token_to_idx.get(tokens, self.unk)
		return [self.__getitem__(token) for token in tokens]

	def to_tokens(self, indices):
		if not isinstance(indices, (list, tuple)):
			return self.idx_to_token[indices]
		return [self.idx_to_token[index] for index in indices]

	@property
	def unk(self):  # 未知词元的索引为0
		return 0


def count_corpus(tokens):
	if len(tokens) == 0 or isinstance(tokens[0], list):
		tokens = [token for line in tokens for token in line]
	return collections.Counter(tokens)


# vocab = Vocab(tokens)
# print('3' * 100)
# print(list(vocab.token_to_idx.items())[:10])
# 整合所有功能
def load_corpus_time_machine(max_tokens=-1):  #@save
    """返回时光机器数据集的词元索引列表和词表"""
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    # 因为时光机器数据集中的每个文本行不一定是一个句子或一个段落，
    # 所以将所有文本行展平到一个列表中
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

corpus, vocab = load_corpus_time_machine()
print(len(corpus), len(vocab))