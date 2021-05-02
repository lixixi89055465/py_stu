import torch

rnn = torch.nn.RNN(100, 10, num_layers=2)
print(rnn._parameters.keys())

print(rnn.weight_hh_l0.shape)
print(rnn.weight_ih_l0.shape)

print(rnn.weight_hh_l1.shape)
print(rnn.weight_ih_l1.shape)

x = torch.randn(10, 3, 100)
out, h = rnn(x, torch.zeros(2, 3, 10))

print(out.shape, h.shape)
