import torch

rnn = torch.nn.RNN(input_size=100, hidden_size=20, num_layers=1)
print(rnn)

x = torch.randn(10, 3, 100)

out, h = rnn(x, torch.zeros(1, 3, 20))

print(rnn.weight_hh_l0.shape)
print(rnn.weight_ih_l0.shape)
print(rnn.bias_ih_l0.shape)
print(rnn.bias_hh_l0.shape)
print(out.shape,h.shape)
