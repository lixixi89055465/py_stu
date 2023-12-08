import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

a = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], \
				  [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6], \
				  [7, 8, 9], [1, 2, 3], [4, 5, 6], [7, 8, 9]])
b = torch.tensor([44, 55, 66, 44, 55, 66, 44, 55, 66, 44, 55, 66])

train_ids = TensorDataset(a, b)
print('0' * 100)
print(train_ids[0:2])
print('=' * 100)

print('1' * 100)
for x_train, y_label in train_ids:
	print(x_train, y_label)

train_loader = DataLoader(dataset=train_ids, batch_size=2, shuffle=False)
for i, data in enumerate(train_loader, 1):
	x_data, label = data
	print(f'batch:{i},x_data:{x_data}, label:{label}')
