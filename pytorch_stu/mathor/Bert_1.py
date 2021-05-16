import re
import math
import torch
import numpy as np
from random import *
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

text = (
    'Hello, how are you? I am Romeo.\n'  # R
    'Hello, Romeo My name is Juliet. Nice to meet you.\n'  # J
    'Nice meet you too. How are you today?\n'  # R
    'Great. My baseball team won the competition.\n'  # J
    'Oh Congratulations, Juliet\n'  # R
    'Thank you Romeo\n'  # J
    'Where are you going today?\n'  # R
    'I am going shopping. What about you?\n'  # J
    'I am going to visit my grandmother. she is not very well'  # R
)

sentences = re.sub("[.,!?\\-]", '', text.lower()).split('\n')
word_list = list(set(" ".join(sentences).split()))
word2idx = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}
for i, w in enumerate(word_list):
    word2idx[w] = i + 4

idx2word = {i: w for i, w in enumerate(word2idx)}
vocab_size = len(word2idx)
token_list = list()
for sentence in sentences:
    arr = [word2idx[s] for s in sentence.split()]
    token_list.append(arr)

print(token_list)

maxlen = 30
batch_size = 6
max_pred = 5
n_layers = 6
n_heads = 12
d_model = 768
d_ff = 768 * 4
d_k = d_v = 64
n_segments = 2


# sample preprocess
def max_data():
    batch = []
    positive = negative = 0
    while positive != batch_size / 2 or negative != batch_size / 2:
        tokens_a_index, tokens_b_index = randrange(len(sentences)), randrange(len(sentences))  # sample
        token_a, token_b = token_list[tokens_a_index], token_list[tokens_b_index]
        input_idx = [word2idx['[CLS]']] + token_a + [word2idx['[SEP]']] + token_b + [word2idx['[SEP]']]
        segment_idx = [0] * (1 + len(token_a) + 1) + [1] * (len(token_b) + 1)
        # MASK LM
        n_pred = min(max_pred, max(1, int(len(input_idx) * 0.15)))
        cand_maked_pos = [i for i, token in enumerate(input_idx) if
                          token != word2idx['[CLS]'] and token != word2idx['[SEP]']]
        shuffle(cand_maked_pos)
        masked_tokens, masked_pos = [], []
        for pos in cand_maked_pos[:n_pred]:
            masked_pos.append(pos)
            masked_tokens.append(input_ids[pos])
            if random() < 0.8:
                input_ids[pos] = word2idx['[MASK]']
            elif random() > 0.9:
                index = randint(0, vocab_size - 1)
                while index < 4:
                    index = randint(0, vocab_size - 1)
                input_ids[pos] = index  # replace
        # Zero Paddings
        n_pad = maxlen - len(input_idx)
        input_idx.extend([0] * n_pad)
        segment_idx.extend([0] * n_pad)

        # Zero Padding (100%-15%) tokens
        if max_pred > n_pred:
            n_pad = max_pred - n_pred
            masked_tokens.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)
        if tokens_a_index + 1 == tokens_b_index and positive < batch_size / 2:
            batch.append([input_idx, segment_idx, masked_tokens, masked_pos, True])
            positive += 1


class BERT(nn.Module):
    pass


max_data()
loader = Data.DataLoader()
batch = []
input_ids, segment_ids, masked_tokens, masked_pos, isNext = batch[0]

model = BERT()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adadelta(model.parameters(), lr=0.001)

for epoch in range(180):
    for input_ids, segment_ids, masked_tokens, masked_pos, isNext in loader:
        logits_lm, logits_clsf = model(input_ids, segment_ids, masked_pos)
        loss_lm = criterion(logits_lm.view(-1, vocab_size),
                            masked_tokens.view(-1))
