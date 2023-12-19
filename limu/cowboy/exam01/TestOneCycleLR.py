import torch
from torchvision.models import AlexNet
import matplotlib.pyplot as plt

steps = []
lrs = []
model = AlexNet(num_classes=2)
lr = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.9, \
												total_steps=100)
for epoch in range(10):
	for batch in range(10):
		scheduler.step()
		lrs.append(scheduler.get_last_lr()[0])
		steps.append(epoch * 10 + batch)

plt.figure()
plt.legend()
plt.plot(steps, lrs, label='OneCycle')
plt.show()
