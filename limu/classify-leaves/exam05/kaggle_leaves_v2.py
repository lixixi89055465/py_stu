import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
import os
import cv2
import timm

import albumentations
from albumentations import pytorch as AT

# below are all from https://github.com/seefun/TorchUtils, thanks seefun to provide such useful tools
# import torch_utils as tu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'device:{device}')
seed = 415
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

CLEAN_DATASET = 0
path = '../data/'
FOLD = 5
csv = pd.read_csv(os.path.join(path, 'clean_train_v4.csv')) \
	if CLEAN_DATASET else pd.read_csv(os.path.join(path, 'train.csv'))
sfolder = StratifiedKFold(n_splits=FOLD, random_state=seed, shuffle=True)
train_folds = []
val_folds = []

for train_idx, val_idx in sfolder.split(csv['image'], csv['label']):
	train_folds.append(train_idx)
	val_folds.append(val_idx)
# print(len(train_idx), len(val_idx))

labelmap_list = sorted(list(set(csv['label'])))
labelmap = dict()
for i, label in enumerate(labelmap_list):
	labelmap[label] = i


# print(labelmap)


class LeavesDataset(torch.utils.data.Dataset):
	def __init__(self, csv, transform=None):
		self.csv = csv
		self.transform = transform
		self.leavelen = len(self.csv)

	def __len__(self):
		return self.leavelen

	def __getitem__(self, idx):
		img = cv2.imread(os.path.join(path, self.csv['image'][idx]))
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		label = labelmap[self.csv['label'][idx]]
		if self.transform:
			img = self.transform(image=img)['image']
		return img, torch.tensor(label).type(torch.LongTensor)


def create_dls(train_csv, test_csv, train_transform, test_transform, bs, num_workers):
	train_ds = LeavesDataset(train_csv, train_transform)
	test_ds = LeavesDataset(test_csv, test_transform)
	train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, \
						  num_workers=num_workers, pin_memory=True, drop_last=True)
	test_dl = DataLoader(test_ds, batch_size=bs, shuffle=False, \
						 num_workers=num_workers, pin_memory=True, drop_last=False)
	return train_dl, test_dl, len(train_ds), len(test_ds)


train_transform1 = albumentations.Compose([
	albumentations.Resize(112, 112, interpolation=cv2.INTER_AREA),
	albumentations.RandomRotate90(p=0.5),
	albumentations.Transpose(p=0.5),
	albumentations.Flip(p=0.5),
	albumentations.ShiftScaleRotate(shift_limit=0.0625, \
									scale_limit=0.0625, \
									rotate_limit=45, \
									border_mode=1, p=0.5),
	albumentations.Normalize(),
	# transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
	AT.ToTensorV2(),
])

test_transform1 = albumentations.Compose([
	albumentations.Resize(112, 112, interpolation=cv2.INTER_AREA),
	albumentations.Normalize(),
	AT.ToTensorV2(),
])


class LeavesTestDataset(Dataset):
	def __init__(self, csv, transform=None):
		self.csv = csv
		self.transform = transform

	def __len__(self):
		return len(self.csv['image'])

	def __getitem__(self, idx):
		img = Image.open(self.csv['image'][idx])
		if self.transform:
			img = self.transform(img)
		return img


def create_testdls(test_csv, test_transform, bs):
	test_ds = LeavesTestDataset(test_csv, test_transform)
	test_dl = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=2)
	return test_dl


transform_test = transforms.Compose([
	transforms.Resize((112, 112)),
	transforms.ToTensor(),
	transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def show_img(x):
	trans = transforms.ToPILImage()
	mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
	std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
	x = (x * std) + mean
	x_pil = trans(x)
	return x_pil


train_csv = csv.iloc[train_folds[0]].reset_index()
val_csv = csv.iloc[val_folds[0]].reset_index()
train_dl, val_dl, n_train, n_val = create_dls(train_csv, \
											  val_csv, \
											  train_transform=train_transform1,
											  test_transform=test_transform1, \
											  bs=64, num_workers=4)


class LabelSmoothing(nn.Module):
	'''
	NLL loss with label smoothing
	'''

	def __init__(self, smoothing=0.0):
		super(LabelSmoothing, self).__init__()
		self.confidence = 1. - smoothing
		self.smoothing = smoothing

	def forward(self, x, target):
		logprobs = torch.nn.functional.softmax(x, dim=-1)
		nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
		nll_loss = nll_loss.squeeze(1)
		smooth_loss = -logprobs.mean(dim=-1)
		loss = self.confidence * nll_loss + self.smoothing * smooth_loss
		return loss.mean()


def find_lr(model, factor, train_dl, optimizer, loss_fn, \
			device, init_lr=1e-8, \
			final_lr=1e-1, beta=0.98, plot=True, save_dir=None):
	num = len(train_dl) - 1
	mult = (find_lr / init_lr) ** (1 / num)
	lr = init_lr
	optimizer.param_groups[0]['lr'] = lr
	avg_loss = 0.0
	best_loss = 0.
	batch_num = 0
	losses = []
	log_lrs = []
	scaler = torch.cuda.amp.GradScaler()  # for AMP training
	if 1:
		for x, y in train_dl:
			x, y = x.to(device), y.to(device)
			batch_num += 1
			optimizer.zero_grad()
			with torch.cuda.amp.autocast():
				out = model(x)
				loss = loss_fn(out, y)
			# smoothen the loss
			avg_loss = beta * avg_loss + \
					   (1 - beta) * loss.data.item()
			smoothed_loss = avg_loss / (1 - beta ** batch_num)  # bias correction
			# stop if loss explodes
			if batch_num > 1 and smoothed_loss > 4 * best_loss:  # prevents explosion
				break
			if smoothed_loss < best_loss or batch_num == 1:
				best_loss = smoothed_loss
			# record the best loss
			losses.append(smoothed_loss)
			log_lrs.append(math.log10(lr))
			# do the sgd step
			# loss.backward()
			# optimizer.step()
			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()
			# update the lr for the next step
			lr *= mult
			optimizer.param_groups[0]['lr'] = lr
	# Suggest a learning rate
	log_lrs, losses = np.array(log_lrs), np.array(losses)
	idx_min = np.argmin(losses)
	min_log_lr = log_lrs[idx_min]
	lr_auto = (10 ** (min_log_lr)) / factor
	if plot:
		selected = [np.argmin(np.abs(log_lrs - (min_log_lr - 1)))]  # high lights the suggsted lr
		plt.figure()
		plt.plot(log_lrs, losses, '-gD', markevery=selected)
		plt.xlabel('log_lrs')
		plt.ylabel('loss')
		plt.title('LR Range Test')
		if save_dir is not None:
			plt.savefig(f'{save_dir}/lr_range_test.png')
		else:
			plt.savefig(f'lr_range_test.png')
	return lr_auto


def get_learner(lr, nb, epochs, model_name='resnet50d', MIXUP=0.1):
	pass


if __name__ == '__main__':
	for x, y in train_dl:
		# imgs_train, labels_train = mixup_fn(x, y)
		break
	# show_img((x[2]))
	print(x.shape)

	model = timm.create_model(
		"resnet50d",
		pretrained=True,
		pretrained_cfg_overlay=dict(file='../../classify-leaves/pth/resnet50d.pth')
	)

	model.fc = nn.Linear(model.fc.in_features, len(labelmap_list))
	nn.init.xavier_normal_(model.fc.weight)
	model = model.to(device)

	print('end' * 100)

	params_1x = [param for name, param in model.named_parameters()
				 if name not in ['fc.weight', 'fc.bias']]
	lr = 5e-4
	optimizer = torch.optim.AdamW([{'params': params_1x},
								   {'params': model.fc.parameters(),
									'lr': lr * 10}],
								  lr=lr, weight_decay=1e-3)  # finetuning
	'''
	from optim import RangerLars
	optimizer = RangerLars([{'params': params_1x},
	                        {'params': model.fc.parameters(),
	                                    'lr': lr * 10}], lr=lr, weight_decay=0.001)
	'''
	# optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=0.001)

	loss_fn = LabelSmoothing(0.1)
	import math
	import matplotlib.pyplot as plt
	import numpy as np
# lr_suggested=find_lr(model,100,train_dl,optimizer,loss_fn,'cuda',init_lr=1e-10,\
# 					 find_lr=1.)#run if u want suggestion from autolr
# lr_suggested
