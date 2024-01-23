# -*- coding: utf-8 -*-
# @Time : 2024/1/23 16:32
# @Author : nanji
# @Site : 
# @File : 10_4.py
# @Software: PyCharm 
# @Comment : 
import os
import torch
from torch import nn
from d2l import torch as d2l


class AttentionDecoder(d2l.Decoder):
	def __init__(self, **kwargs):
		super(AttentionDecoder, self).__init__(**kwargs)

	@property
	def attention_weights(self):
		raise NotImplementedError


class Seq2SeqAttentionDecoder(AttentionDecoder):
	def __init__(self, vocab_size, embed_size, num_hiddens, \
				 num_layers, dropout=0, **kwargs):
		super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
		self.attention = d2l.AdditiveAttention(num_hiddens, \
											   num_hiddens, \
											   num_hiddens, \
											   dropout)
		self.embedding = nn.Embedding(vocab_size, embed_size)

		self.rnn = nn.GRU(embed_size + num_hiddens, \
						  num_hiddens, \
						  num_layers, \
						  dropout)
		self.dense = nn.Linear(num_hiddens, vocab_size)

	def init_state(self, enc_outputs, enc_valid_lens, *args):
		outputs, hidden_state = enc_outputs
		return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)

	def forward(self, X, state):
		enc_outputs, hidden_state, enc_valid_lens = state
		X = self.embedding(X).permute(1, 0, 2)
		outputs, self._attention_weights = [], []
		for x in X:
			query = torch.unsqueeze(hidden_state[-1], dim=1)
			context = self.attention(
				query, enc_outputs, enc_outputs, enc_valid_lens
			)
			x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
			out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
			outputs.append(out)
			self._attention_weights.append(self.attention.attention_weights)
		outputs = self.dense(torch.cat(outputs, dim=0))
		return outputs.permute(1, 0, 2), \
			   [enc_outputs, hidden_state, enc_valid_lens]

	@property
	def attention_weights(self):
		return self._attention_weights


encoder = d2l.Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16, \
							 num_layers=2)
encoder.eval()
decoder = Seq2SeqAttentionDecoder(vocab_size=10, embed_size=8, num_hiddens=16, \
								  num_layers=2)
decoder.eval()
X = torch.zeros((4, 7), dtype=torch.long)
state = decoder.init_state(encoder(X), None)
output, state = decoder(X, state)
print('5' * 100)
print(output.shape, len(state), state[0].shape, len(state[1]), state[1][0].shape)
