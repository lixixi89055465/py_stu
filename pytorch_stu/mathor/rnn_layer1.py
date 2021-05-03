import torch

rnn=torch.nn.RNN(100,20)
print(rnn._parameters.keys())
print(rnn.weight_ih_l0.shape)
print(rnn.weight_hh_l0.shape)
print(rnn.bias_ih_l0.shape)
print(rnn.bias_hh_l0.shape)
rnn = torch.nn.RNN(input_size=100, hidden_size=20, num_layers=7)
x = torch.randn(10, 3, 100)
out, h_t = rnn(x, torch.zeros(7, 3, 20))
print(out.shape)
print(h_t.shape)
