import torch

rnn = torch.nn.GRU(100, 10,2)
print(rnn._parameters.keys())

print(rnn.weight_hh_l0.shape)
print(rnn.weight_ih_l0.shape)
print(rnn.bias_ih_l0.shape)
print(rnn.bias_hh_l0.shape)
