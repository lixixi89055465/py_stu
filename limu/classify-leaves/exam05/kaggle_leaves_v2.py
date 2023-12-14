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
	print(len(train_idx), len(val_idx))

labelmap_list = sorted(list(set(csv['label'])))
labelmap = dict()
for i, label in enumerate(labelmap_list):
	labelmap[label] = i

print(labelmap)


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
for x, y in train_dl:
	# imgs_train, labels_train = mixup_fn(x, y)
	break
print('6' * 100)
print(y.shape)
print('7' * 100)
show_img((x[2]))

print(x.shape)