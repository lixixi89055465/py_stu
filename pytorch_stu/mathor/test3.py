import torch
import torch.nn as nn

lstm = nn.LSTM(input_size=100, hidden_size=20, num_layers=4)
x = torch.randn(10, 3, 100) # 一个句子10个单词，送进去3条句子，每个单词用一个100维的vector表示
out, (h, c) = lstm(x)
print(out.shape, h.shape, c.shape)
# torch.Size([10, 3, 20]) torch.Size([4, 3, 20]) torch.Size([4, 3, 20])