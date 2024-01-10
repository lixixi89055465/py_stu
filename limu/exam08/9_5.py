# -*- coding: utf-8 -*-
# @Time : 2024/1/10 10:07
# @Author : nanji
# @Site : 
# @File : 9_5.py
# @Software: PyCharm 
# @Comment : 
import os
import torch
from torch import nn
from d2l import torch as d2l

import time

start = time.time()
# @save
d2l.DATA_HUB['fra-eng'] = (d2l.DATA_URL + 'fra-eng.zip',
						   '94646ad1522d915e7b0f9296181140edcf86a4f5')


# @save
def read_data_nmt():
	"""载入“英语－法语”数据集"""
	data_dir = d2l.download_extract('fra-eng')
	with open(os.path.join(data_dir, 'fra.txt'), 'r',
			  encoding='utf-8') as f:
		return f.read()

raw_text = read_data_nmt()

def preprocess_nmt(text):
	def no_space(char, prev_char):
		return char in set(',.!?') and prev_char != ' '

	text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
	out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
		   for i, char in enumerate(text)]
	return ''.join(out)


def tokenize_nmt(text, num_examples=None):
	source, target = [], []
	for i, line in enumerate(text.split('\n')):
		if num_examples and i > num_examples:
			break
		parts = line.split('\t')
		if len(parts) == 2:
			source.append(parts[0].split(' '))
			source.append(parts[1].split(' '))
	return source, target


import pickle

fname = 'my.txt'
if not os.path.exists(fname):
	text = preprocess_nmt(raw_text)
	with open(fname, 'wb') as f:
		pickle.dump(text, f)
with open(fname, 'rb') as f:
	text = pickle.load(f)
fsource = 'source.txt'
if not os.path.exists(fsource):
	source, target = tokenize_nmt(text)
	with open(fsource, 'wb') as f:
		pickle.dump((source, target), f)
with open(fsource, 'rb') as f:
	(source, target) = pickle.load(f)

print(source[:6], target[:6])

#@save
def show_list_len_pair_hist(legend, xlabel, ylabel, xlist, ylist):
    """绘制列表长度对的直方图"""
    d2l.set_figsize()
    _, _, patches = d2l.plt.hist(
        [[len(l) for l in xlist], [len(l) for l in ylist]])
    d2l.plt.xlabel(xlabel)
    d2l.plt.ylabel(ylabel)
    for patch in patches[1].patches:
        patch.set_hatch('/')
    d2l.plt.legend(legend)

show_list_len_pair_hist(['source', 'target'], '# tokens per sequence',
                        'count', source, target);


def show_list_len_pair_hist(legend, xlabel, ylabel, xlist, ylist):
    """绘制列表长度对的直方图"""
    d2l.set_figsize()
    _, _, patches = d2l.plt.hist(
        [[len(l) for l in xlist], [len(l) for l in ylist]])
    d2l.plt.xlabel(xlabel)
    d2l.plt.ylabel(ylabel)
    for patch in patches[1].patches:
        patch.set_hatch('/')
    d2l.plt.legend(legend)

show_list_len_pair_hist(['source', 'target'], '# tokens per sequence',
                        'count', source, target);
src_vocab = d2l.Vocab(source, min_freq=2,
                      reserved_tokens=['<pad>', '<bos>', '<eos>'])
len(src_vocab)
#@save
def truncate_pad(line, num_steps, padding_token):
    """截断或填充文本序列"""
    if len(line) > num_steps:
        return line[:num_steps]  # 截断
    return line + [padding_token] * (num_steps - len(line))  # 填充

truncate_pad(src_vocab[source[0]], 10, src_vocab['<pad>'])
#@save
def truncate_pad(line, num_steps, padding_token):
    """截断或填充文本序列"""
    if len(line) > num_steps:
        return line[:num_steps]  # 截断
    return line + [padding_token] * (num_steps - len(line))  # 填充

truncate_pad(src_vocab[source[0]], 10, src_vocab['<pad>'])

#@save
def build_array_nmt(lines, vocab, num_steps):
    """将机器翻译的文本序列转换成小批量"""
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = torch.tensor([truncate_pad(
        l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_len


#@save
def load_data_nmt(batch_size, num_steps, num_examples=600):
    """返回翻译数据集的迭代器和词表"""
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text, num_examples)
    src_vocab = d2l.Vocab(source, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = d2l.Vocab(target, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = d2l.load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab

train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=8)
for X, X_valid_len, Y, Y_valid_len in train_iter:
    print('X:', X.type(torch.int32))
    print('X的有效长度:', X_valid_len)
    print('Y:', Y.type(torch.int32))
    print('Y的有效长度:', Y_valid_len)
    break
