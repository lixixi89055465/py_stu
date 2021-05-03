import torch

lstm = torch.nn.LSTM(input_size=100, hidden_size=20, num_layers=1)
print(lstm._all_weights)
x = torch.randn(10, 3, 100)
out, (h, c) = lstm(x)
print(out.shape)
print(h.shape)
print(c.shape)
