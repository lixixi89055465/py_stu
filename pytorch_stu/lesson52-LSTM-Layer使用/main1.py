import torch

lstm = torch.nn.LSTM(input_size=100, hidden_size=20, num_layers=4)
print(lstm)

x = torch.randn(10, 3, 100)
out, (h, c) = lstm(x)
print(out.shape, h.shape, c.shape)
print(out.shape)
