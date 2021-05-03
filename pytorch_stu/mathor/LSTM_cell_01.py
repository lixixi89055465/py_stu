import torch
import torch.nn as nn

cell = torch.nn.LSTMCell(input_size=100, hidden_size=20)
print(cell)
h = torch.zeros(3, 20)
c = torch.zeros(3, 20)
x = torch.randn(10, 3, 100)
for xt in x:
    h, c = cell(xt, [h, c])

print(h.shape, c.shape)
