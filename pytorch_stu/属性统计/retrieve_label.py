import torch

prob = torch.randn(4, 10)
print(prob)
idx = prob.topk(dim=1, k=3)
print(idx)
idx = idx[1]
print('idx:')
print(idx)
label = torch.arange(10) + 100
print(label)

print('-' * 20)
print(label.expand(4, 10))
# 根据索引集合和维度获取对应数据
a = torch.gather(label.expand(4, 10), dim=1, index=idx.long())
print(a)
