import torch

x = torch.randn(1, 784)
w = torch.randn(10, 784)
logits = x @ w.t()
print(logits.shape)

pred = torch.nn.functional.softmax(logits, dim=1)
print(pred)
pred_log=torch.log(pred)
print(torch.nn.functional.cross_entropy(logits, torch.tensor([3])))

print("pred_log:")
print(pred_log)
print(torch.nn.functional.nll_loss(pred_log, torch.tensor([3])))

print(torch.tensor([3]).shape)