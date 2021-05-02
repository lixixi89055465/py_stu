import torch
import torch.nn as nn

x = torch.LongTensor([[1, 2, 3, 4, 5, 6], [4, 3, 2, 5, 6, 7]])
embeddings = nn.Embedding(8, 5, padding_idx=4)  # 5个词,每个词也是5维
print(embeddings(x))
print(embeddings(x).size())
