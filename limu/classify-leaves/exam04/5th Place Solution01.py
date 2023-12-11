import numpy as np
import pandas as pd
import os
from PIL import Image
import cv2
import math

import torch
import torchvision
import timm
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold, cross_val_score
# Metric
from sklearn.metrics import f1_score, accuracy_score

# Augmentation
import albumentations
from albumentations.pytorch.transforms import ToTensorV2

if torch.cuda.is_available():
	device = torch.device('cuda')
else:
	device = torch.device('cpu')

print(f'using device {device}')
seed = 415
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

path = '../data/'
labels_file_path = os.path.join(path, 'train.csv')
sample_submission_path = os.path.join(path, 'test.csv')

df = pd.read_csv(labels_file_path)
sub_df = pd.read_csv(sample_submission_path)
print(df.head())
labels_unique = df['label'].unique()
print(len(labels_unique))
le = LabelEncoder()
le.fit(df['label'])
df['label']
le.transform(df['label'])
label_map = dict(zip(le.classes_, le.transform(le.classes_)))

label_inv_map = {v: k for k, v in label_map.item()}


def get_train_transforms():
	return albumentations.Compose([
		albumentations.Resize(320, 320),
		albumentations.HorizontalFlip(p=0.5),
		albumentations.VerticalFlip(p=0.5),
		albumentations.Rotate(limit=180, p=0.7),
		albumentations.RandomBrightnessContrast(),
		albumentations.ShiftScaleRotate(
			shift_limit=.25, scale_limit=.1, rotate_limit=0
		),
		albumentations.Normalize(
			[0.485, 0.456, 0.406], [0.229, 0.224, 0.225],
			max_pixel_value=255.0, always_apply=True
		),
		ToTensorV2(p=1.)
	])


def get_valid_transforms():
	return albumentations.Compose([
		albumentations.Resize(320, 320),
		albumentations.Normalize(
			[0.485, 0.456, 0.406], [0.229, 0.224, 0.225],
			max_pixel_value=255.0, always_apply=True
		),
		ToTensorV2(p=1.)
	])


class LeafDataset(torch.utils.data.Dataset):
	def __init__(self, images_filepaths, labels, transform=None):
		self.images_filepaths = images_filepaths
		self.labels = labels
		self.transform = transform
		self.datalen = len(labels)

	def __len__(self):
		return self.datalen

	def __getitem__(self, idx):
		image_filePath = self.images_filepaths[idx]
		image = cv2.imread(image_filePath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		label = self.labels[idx]
		if self.transform is not None:
			image = self.transform(image=image)['image']
		return image, label


def accuracy(output, target):
	y_pred = torch.softmax(output, dim=-1)
	y_pred = torch.argmax(y_pred, dim=-1).cpu()
	target = target.cpu()
	return accuracy_score(target, y_pred)


def calculate_f1_macro(output, target):
	y_pred = torch.softmax(output, dim=1)
	y_pred = torch.argmax(y_pred, dim=1).cpu()
	target = target.cpu()
	return f1_score(target, y_pred, average='macro')


class MetricMonitor:
	def __init__(self, float_precision=3):
		self.float_precision = float_precision
		self.reset()

	def reset(self):
		self.metrics = defaultdict(lambda: {'val': 0, 'count': 0, 'avg': 0})

	def update(self, metric_name, val):
		metric = self.metrics[metric_name]
		metric['val'] += val
		metric['count'] += 1
		metric['avg'] = metric['val'] / metric['count']

	def __str__(self):
		return " | ".join(
			[
				"{metric_name}: {avg:.{float_precision}f}".format(
					metric_name=metric_name, avg=metric["avg"],
					float_precision=self.float_precision
				)
				for (metric_name, metric) in self.metrics.items()
			]
		)


def calc_learning_rate(epoch, init_lr, n_epochs, batch=0, \
					   nBatch=None, lr_schedule_type='cosine'):
	if lr_schedule_type == 'coisine':
		t_total = n_epochs * nBatch
		t_cur = epoch * nBatch + batch
		lr = 0.5 * init_lr * (1 + math.cos(math.pi * t_cur / t_total))
	elif lr_schedule_type is None:
		lr = init_lr
	else:
		raise ValueError('do not support: %s' % lr_schedule_type)
	return lr


def adjust_learning_rate(optimizer, epoch, params, batch=0, nBatch=None):
	new_lr = calc_learning_rate(epoch, params['lr'], params['epochs'], batch, nBatch)
	for param_group in optimizer.param_groups():
		param_group['lr'] = new_lr
	return new_lr

params={
	'model': 'seresnext50_32x4d',
	# 'model': 'resnet50d',
	'device': device,
	'lr': 1e-3,
	'batch_size': 64,
	'num_workers': 4,
	'epochs': 50,
	'out_features': df['label'].nunique(),
	'weight_decay': 1e-5
}

class LeafNet(nn.Module):
	pass
