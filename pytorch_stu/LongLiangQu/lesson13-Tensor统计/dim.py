import torch
a=torch.rand(4,10)
print(a.shape)
print(a.max(dim=1))

print(a.min(dim=1))
print(a.argmax(dim=1))
# 统计结果保持原来的维度
print(a.max(dim=1, keepdim=True))
print(a.argmax(dim=1, keepdim=True))

