# -*- coding: utf-8 -*-
# @Time : 2024/1/11 17:44
# @Author : nanji
# @Site : 
# @File : 9_7.py
# @Software: PyCharm 
# @Comment : 
import torch
from d2l import torch as d2l
from torch import nn
import math
import collections


class Seq2SeqEncoder(d2l.Encoder):
	def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, \
				 dropout=0, **kwargs):
		super(Seq2SeqEncoder, self).__init__(**kwargs)
		self.embeding = nn.Embedding(vocab_size, embed_size)
		self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, \
						  dropout=dropout)

	def forward(self, X, *args):
		X = self.embeding(X)
		X = X.permute(1, 0, 2)
		output, state = self.rnn(X)
		return output, state


encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16, \
						 num_layers=2)

# encoder.eval()
X = torch.zeros((4, 7), dtype=torch.long)


# output, state = encoder(X)
# print('0'*100)
# print(output.shape)

class Seq2SeqDecoder(d2l.Decoder):
	def __init__(self, vocab_size, embed_size, num_hiddens, \
				 num_layers, dropout=0, **kwargs):
		super(Seq2SeqDecoder, self).__init__(**kwargs)
		self.embedding = nn.Embedding(vocab_size, embed_size)
		self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, \
						  dropout=dropout)
		self.dense = nn.Linear(num_hiddens, vocab_size)

	def init_state(self, enc_output, *args):
		return enc_output[1]

	def forward(self, X, state):
		X = self.embedding(X).permute(1, 0, 2)
		context = state[-1].repeat(X.shape[0], 1, 1)
		X_and_context = torch.cat((X, context), dim=2)
		output, state = self.rnn(X_and_context, state)
		output = self.dense(output).permute(1, 0, 2)
		return output, state


decoder = Seq2SeqDecoder(vocab_size=10, embed_size=8, \
						 num_hiddens=16, num_layers=2)


# decoder.eval()
# state = decoder.init_state(encoder(X))
# output, state = decoder(X, state)
# print('0' * 100)
# print(output.shape, state.shape)


def sequence_mask(X, valid_len, value=0):
	maxlen = X.size(1)
	mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] \
		   < valid_len[:, None]
	X[~mask] = value
	return X


# X = torch.tensor([[1, 2, 3], [4, 5, 6]])
# print(sequence_mask(X, torch.tensor([1, 2])))
# X = torch.ones(2, 3, 4)
# print('2'*100)
# print(sequence_mask(X, torch.tensor([1, 2]), value=-1))
class MaskedSoftMaxCELoss(nn.CrossEntropyLoss):
	def forward(self, pred, label, valid_len):
		weights = torch.ones_like(label)
		weights = sequence_mask(weights, valid_len)
		self.reduction = 'none'
		unweighted_loss = super(MaskedSoftMaxCELoss, self).forward(
			pred.permute(0, 2, 1), label
		)
		weighted_loss = (unweighted_loss * weights).mean(dim=1)
		return weighted_loss


loss = MaskedSoftMaxCELoss()
a = loss(torch.ones(3, 4, 10), torch.ones((3, 4), dtype=torch.long), \
		 torch.tensor([4, 2, 0]))
print(a)


def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
	def xavier_init_weights(m):
		if type(m) == nn.Linear:
			nn.init.xavier_uniform_(m.weight)
		if type(m) == nn.GRU:
			for param in m._flat_weights_names:
				if 'weight' in param:
					nn.init.xavier_uniform_(m._parameters[param])

	net.apply(xavier_init_weights)
	net.to(device)
	optimizer = torch.optim.Adam(net.parameters(), lr=lr)
	loss = MaskedSoftMaxCELoss()
	net.train()
	animator = d2l.Animator(xlabel='epoch', ylabel='loss', \
							xlim=[10, num_epochs])
	for epoch in range(num_epochs):
		timer = d2l.Timer()
		metric = d2l.Accumulator(2)
		for batch in data_iter:
			optimizer.zero_grad()
			X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
			bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0], \
							   device=device).reshape(-1, 1)
			dec_input = torch.cat([bos, Y[:, :-1]], 1)
			Y_hat, _ = net(X, dec_input, X_valid_len)
			L = loss(Y_hat, Y, Y_valid_len)
			L.sum().backward()
			d2l.grad_clipping(net, 1)
			num_tokens = Y_valid_len.sum()
			optimizer.step()
			with torch.no_grad():
				metric.add(L.sum(), num_tokens)
		if (epoch + 1) % 10 == 0:
			animator.add(epoch + 1, (metric[0] / metric[1],))
	print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
		  f'tokens/sec on {str(device)}')


embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
batch_size, num_steps = 64, 10
lr, num_epochs, device = 0.005, 300, d2l.try_gpu(2)
train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(
	batch_size,
	num_steps
)
encoder = Seq2SeqEncoder(len(src_vocab), \
						 embed_size, num_hiddens, num_layers, dropout)
decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, \
						 num_layers, dropout)
net = d2l.EncoderDecoder(encoder, decoder)


# train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
# print('0' * 100)
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, \
					num_steps, device, \
					save_attention_weights=False):
	net.eval()
	src_tokens = src_vocab[src_sentence.lower().split(' ')] + [src_vocab['<eos>']]
	enc_valid_len = torch.tensor([len(src_tokens)])
	src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
	enc_X = torch.unsqueeze(
		torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
	enc_outputs = net.encoder(enc_X, enc_valid_len)
	dec_state = net.decoder(enc_X, enc_valid_len)
	dec_X = torch.unsqueeze(torch.tensor(
		[tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
	output_seq, attention_weight_seq = [], []
	for _ in range(num_steps):
		Y, dec_state = net.decoder(dec_X, dec_state)
		dec_X = Y.argmax(dim=2)
		pred = dec_X.squeeze(dim=0).type(torch.int32).item()
		if save_attention_weights:
			attention_weight_seq.append(net.decoder.attention_weights)
		if pred == tgt_vocab['<eos>']:
			break
		output_seq.append(pred)
	return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq


def bleu(pred_seq, label_seq, k):
	"""计算BLEU"""
	pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
	len_pred, len_label = len(pred_tokens), len(label_tokens)
	score = math.exp(min(0, 1 - len_label / len_pred))
	for n in range(1, k + 1):
		num_matches, label_subs = 0, collections.defaultdict(int)
		for i in range(len_label - n + 1):
			label_subs[' '.join(label_tokens[i: i + n])] += 1
		for i in range(len_pred - n + 1):
			if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
				num_matches += 1
				label_subs[' '.join(pred_tokens[i: i + n])] -= 1
		score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
	return score


engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
	translation, attention_weight_seq = predict_seq2seq(
		net, eng, src_vocab, tgt_vocab, num_steps, device
	)
	print(f'{eng} => {translation} ,bleu {bleu(translation, fra, k=2):.3f}')
