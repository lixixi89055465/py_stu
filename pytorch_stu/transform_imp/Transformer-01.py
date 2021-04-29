'''
  code by Tae Hwan Jung(Jeff Jung) @graykode, Derek Miller @dmmiller612, modify by wmathor
  Reference : https://github.com/jadore801120/attention-is-all-you-need-pytorch
              https://github.com/JayParks/transformer
'''

import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# S: decoding input
# E: decoding output
sentences = [
    # enc_input           dec_input         dec_output
    ['ich mochte ein bier P', 'S i want a beer .', 'i want a beer . E'],
    ['ich mochte ein cola P', 'S i want a coke .', 'i want a coke . E']
]
# Padding Should be Zero
src_vocab = {'P': 0, 'ich': 1, 'want': 1, 'mochte': 2, 'ein': 3, 'bier': 4, 'cola': 5}
src_vocab_size = len(src_vocab)
tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'coke': 5, 'S': 6, 'E': 7, '.': 8}
tgt_vocab_size = len(tgt_vocab)
print(src_vocab_size, tgt_vocab_size)

idx2word = {i: w for i, w in enumerate(tgt_vocab)}

src_len = 5
tgt_len = 6

# Transformer Parameter
d_model = 512  # Embedding Size
d_ff = 2048  # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q),V
n_layers = 6  # number of Encoder of Decoder Layer
n_heads = 8  # number of heads in Multi-Head Attention


def make_data(sentences):
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for i in range(len(sentences)):
        enc_input = [[src_vocab[n] for n in sentences[i][0].split()]]  # [[1,2,3,4,0],[1,2,3,5,0]]
        dec_input = [[tgt_vocab[n] for n in sentences[i][1].split()]]  # [[6,1,2,3,4,8],[6,1,2,3,5,8]]
        dec_output = [[tgt_vocab[n] for n in sentences[i][2].split()]]  # [[1,2,3,4,8,7],[1,2,3,5,8,7]]
        enc_inputs.extend(enc_input)
        dec_inputs.extend(dec_input)
        dec_outputs.extend(dec_output)
    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)


enc_inputs, dec_inputs, dec_outputs = make_data(sentences)


class MyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]


loader = torch.utils.data.DataLoader(MyDataset(enc_inputs, dec_inputs, dec_outputs), 2, True)

