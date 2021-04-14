import torch

a1 = torch.ones(128, 39,dtype=torch.float32)
b1 = torch.ones(128,dtype=torch.int64)
loss1=torch.nn.CrossEntropyLoss()
print(loss1(a1, b1))
