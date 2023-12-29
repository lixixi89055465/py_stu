# -*- coding: utf-8 -*-
# @Time : 2023/12/29 9:55
# @Author : nanji
# @Site : 
# @File : 8_3.py
# @Software: PyCharm 
# @Comment : 
import os
import torch
from torch import nn
from d2l import torch as d2l
import random

tokens = d2l.tokenize(d2l.read_time_machine())
# 因为每个文本行不一定是一个句子或一个段落，因此我们把所有文本行拼接到一起
corpus = [token for line in tokens for token in line]
vocab = d2l.Vocab(corpus)
print(vocab.token_freqs[:10])

freqs = [freq for token, freq in vocab.token_freqs]
d2l.plot(freqs, xlabel='token:x', ylabel='frequency:n(X)', \
		 xscale='log', yscale='log')

bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
bigram_vocab = d2l.Vocab(bigram_tokens)
print('0' * 100)
print(bigram_vocab.token_freqs[:10])
trigram_tokens = [triple for triple in zip(corpus[:-2], corpus[1:-1], corpus[2:])]
trigram_vocab = d2l.Vocab(trigram_tokens)
print('1' * 100)
print(trigram_vocab.token_freqs[:10])
bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]


# d2l.plot([freqs, bigram_freqs, trigram_freqs], \
# 		 xlabel='token:x', ylabel='frequency : n(x) ', \
# 		 xscale='log', yscale='log', \
# 		 legend=['unigram', 'bigram', 'trigram'])
def seq_data_iter_random(corpus, batch_size, num_steps):
	corpus = corpus[random.randint(0, num_steps - 1):]
	num_subseqs = (len(corpus) - 1) // num_steps
	initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
	random.shuffle(initial_indices)

	def data(pos):
		return corpus[pos:pos + num_steps]

	num_batches = num_subseqs // batch_size
	for i in range(0, batch_size * num_batches, batch_size):
		initial_indices_per_batch = initial_indices[i:i + batch_size]
		X = [data(j) for j in initial_indices_per_batch]
		Y = [data(j + 1) for j in initial_indices_per_batch]
		yield torch.tensor(X), torch.tensor(Y)


def seq_data_iter_random(corpus, batch_size, num_steps):
	corpus = corpus[random.randint(num_steps - 1):]
	num_subseqs = (len(corpus) - 1) // num_steps
	initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
	random.shuffle(initial_indices)

	def data(pos):
		return corpus[pos:pos + num_steps]

	num_batches = num_subseqs // batch_size
	for i in range(0, batch_size * num_steps, batch_size):
		initial_indices_per_batch = initial_indices[i:i + batch_size]
		X = [data(j) for j in initial_indices_per_batch]
		Y = [data(j + 1) for j in initial_indices_per_batch]
		yield torch.tensor(X), torch.tensor(Y)


my_seq = list(range(35))


# for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
# 	print('X: ', X, '\nY:', Y)

def seq_data_iter_sequential(corpus, batch_size, num_steps):
	'''使用顺序分区生成一个小批量子序列 '''
	offset = random.randint(0, num_steps)
	num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
	Xs = torch.tensor(corpus[offset:offset + num_tokens])
	Ys = torch.tensor(corpus[offset + 1:offset + num_tokens + 1])
	Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
	num_batches = Xs.shape[-1] // num_steps
	for i in range(0, num_steps * num_batches, num_steps):
		X = Xs[:, i:i + num_steps]
		Y = Ys[:, i:i + num_steps]
		yield X, Y


# for X, Y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
# 	print('X: ', X, '\nY:', Y)
class SeqDataLoader:
	def __init__(self, batch_size, num_steps, use_random_iter, \
				 max_tokens):
		if use_random_iter:
			self.data_iter_fn = d2l.seq_data_iter_random
		else:
			self.data_iter_fn = d2l.seq_data_iter_sequential
		self.corpus, self.vocab = d2l.load_corpus_time_machine(max_tokens)
		self.batch_size, self.num_steps = batch_size, num_steps

	def __iter__(self):
		return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)


def load_data_time_machine(batch_size, num_steps, \
						   use_random_iter=False, max_tokens=10000):
	data_iter = SeqDataLoader(
		batch_size, num_steps, use_random_iter, max_tokens
	)
	return data_iter, data_iter.vocab
