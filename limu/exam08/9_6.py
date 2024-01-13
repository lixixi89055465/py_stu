# -*- coding: utf-8 -*-
# @Time : 2024/1/11 17:34
# @Author : nanji
# @Site : 
# @File : 9_6.py
# @Software: PyCharm 
# @Comment :

from torch import nn


class Encoder(nn.Module):
	def __init__(self, **kwargs):
		super(Encoder, self).__init__(**kwargs)

	def forward(self, X, *args):
		raise NotImplementedError


class Decoder(nn.Module):
	def __init__(self, **kwargs):
		super(Decoder, self).__init__(**kwargs)

	def init_state(self, enc_outputs, *args):
		raise NotImplementedError

	def forward(self, X, state):
		raise NotImplementedError


class EncoderDecoder(nn.Module):
	def __init__(self, encoder, decoder, **kwargs):
		super(EncoderDecoder, self).__init__(**kwargs)
		self.ecoder = encoder
		self.decoder = decoder

	def forward(self, enc_X, dec_X, *args):
		enc_outputs = self.encoder(enc_X, *args)
		dec_state = self.decoder.init_state(enc_outputs, *args)
		return self.decoder(dec_X, dec_state)
