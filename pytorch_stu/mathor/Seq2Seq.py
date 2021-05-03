import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

letter = [c for c in 'SE?abcdefghijklmnopqrstuvwxyz']
print(letter)

letter2idx = {n: i for i, n in enumerate(letter)}
print(letter2idx)
seq_data = [['man', 'women'], ['black', 'white'], ['king', 'queen'], ['girl', 'boy'], ['up', 'down'], ['high', 'low']]
# seq2seq parameter

n_step = max([max(len(i), len(j)) for i, j in seq_data])
print(n_step)
n_hidden = 128
n_class = len(letter2idx)
batch_size = 3
print(n_class)


def make_data(seq_data):
    enc_input_all, dec_input_all, dec_output_all = [], [], []
    for seq in seq_data:
        for i in range(2):
            seq[i] = seq[i] + '?' * (n_step - len(seq[i]))
        enc_input = [letter2idx[n] for n in (seq[0] + 'E')]
        dec_input = [letter2idx[n] for n in ('S' + seq[1])]
        dec_output = [letter2idx[n] for n in (seq[1] + 'E')]
        enc_input_all.append(np.eye(n_class)[enc_input])
        dec_input_all.append(np.eye(n_class)[dec_input])
        dec_output_all.append(dec_output)
    return torch.Tensor(enc_input_all), torch.Tensor(dec_input_all),torch.LongTensor(dec_output_all)


enc_input_all, dec_input_all, dec_output_all = make_data(seq_data)
print(enc_input_all.shape)
print(dec_input_all.shape)
print(dec_output_all.shape)
print("")

