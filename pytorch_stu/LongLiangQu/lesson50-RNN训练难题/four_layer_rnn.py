import torch

rnn = torch.nn.RNN(input_size=100, hidden_size=20, num_layers=4)
print(rnn)

x = torch.randn(10, 3, 100)
out, h = rnn(x)
print(out.shape, h.shape)
