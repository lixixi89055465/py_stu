import torch

print('one layer lstm')
x = torch.randn(10, 3, 100)
cell = torch.nn.LSTMCell(input_size=100, hidden_size=20)
h = torch.zeros(3, 20)
c = torch.zeros(3, 20)
for xt in x:
    h, c = cell(xt, [h, c])

print(h.shape,c.shape)

