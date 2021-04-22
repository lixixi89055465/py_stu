import torch

print('two layer lstm')

cell1 = torch.nn.LSTMCell(input_size=100, hidden_size=30)
cell2 = torch.nn.LSTMCell(input_size=30, hidden_size=20)
h1 = torch.zeros(3, 30)
c1 = torch.zeros(3, 30)
h2 = torch.zeros(3, 20)
c2 = torch.zeros(3, 20)

x = torch.randn(10, 3, 100)
for xt in x:
    h1, c1 = cell1(xt, [h1, c1])
    h2, c2 = cell2(h1, [h2, c2])

print(h2.shape,c2.shape)

