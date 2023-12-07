import torch
from torch.utils.data import random_split

dataset = range(10)
a = torch.random.seed()
train_dataset, test_dataset = \
	random_split(dataset, lengths=[7, 3],
				 generator=torch.Generator().manual_seed(a))

print(list(train_dataset))
print(list(test_dataset))

torch.manual_seed(0)
train_dataset, test_dataset = random_split(
	dataset=dataset,
	lengths=[7, 3]
)
print(list(train_dataset))
print(list(test_dataset))
