import torch
import torch.nn as nn
from torch.nn import functional as F
import ttach as tta
from resnest.torch import resnest50
from cutmix.cutmix import CutMix
from cutmix.utils import CutMixCrossEntropyLoss

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset

from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import KFold
from PIL import Image
import os
import matplotlib.pyplot as plt
import torchvision.models as models

from tqdm import tqdm

devices = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(devices)
labels_dataframe = pd.read_csv('../data/train.csv')
# print(labels_dataframe.head(5))

leaves_labels = sorted(set(labels_dataframe['label']))
n_classes = len(leaves_labels)
print(n_classes)
print('0' * 100)
print(leaves_labels[:10])
class_to_num = dict(zip(leaves_labels, range(n_classes)))
print('1' * 100)
print(class_to_num)
# 再转换回来，方便最后预测的时候使用
print('2' * 100)
num_to_class = {v: k for k, v in class_to_num.items()}
print(num_to_class)


# 加成pytorch的dataset
class TrainValidData(Dataset):
	def __init__(self, csv_path, file_path, resize_height=224, resize_width=224, transform=None):
		"""
		 Args:
		     csv_path (string): csv 文件路径
		     img_path (string): 图像文件所在路径
		 """
		# 需要调整后的照片尺寸，我这里每张图片的大小尺寸不一致#
		self.resize_height = resize_height
		self.resize_width = resize_width
		self.file_path = file_path
		self.to_tensor = transforms.ToTensor()  # 将数据转换成tensor形式
		self.transform = transform
		# 读取 csv 文件
		# 利用pandas读取csv文件
		self.data_info = pd.read_csv(csv_path, header=None)  # header=None是去掉表头部分
		# 文件第一列包含图像文件名称
		self.image_arr = np.asarray(self.data_info.iloc[1:, 0])  # self.data_info.iloc[1:,0]表示读取第一列，从第二行开始一直读取到最后一行
		# 第二列是图像的 label
		self.label_arr = np.asarray(self.data_info.iloc[1:, 1])
		# 计算 length
		self.data_len = len(self.data_info.index) - 1

	def __getitem__(self, index):
		# 从 image_arr中得到索引对应的文件名
		single_image_name = self.image_arr[index]
		img_as_img = Image.open(self.file_path + single_image_name)
		# 如果需要将RGB三通道的图片转换成灰度图片可参考下面两行
		# if img_as_img.mode != 'L':
		#     img_as_img = img_as_img.convert('L')
		transform = transforms.Compose([
			transforms.Resize((224, 224)),
			transforms.ToTensor()
		])
		# 得到图像的 label
		label = self.label_arr[index]
		number_label = class_to_num[label]
		return (img_as_img, number_label)

	def __len__(self):
		return self.data_len


# 加成pytorch的dataset
class TestData(Dataset):
	def __init__(self, csv_path, file_path, \
				 resize_height=224, resize_width=224, transform=None):
		"""
		 Args:
		     csv_path (string): csv 文件路径
		     img_path (string): 图像文件所在路径
		 """
		# 需要调整后的照片尺寸，我这里每张图片的大小尺寸不一致#
		self.resize_height = resize_height
		self.resize_width = resize_width
		self.file_path = file_path
		self.to_tensor = transforms.ToTensor()  # 将数据转换成tensor形式
		self.transform = transform
		# 读取 csv 文件
		# 利用pandas读取csv文件
		self.data_info = pd.read_csv(csv_path, header=None)  # header=None是去掉表头部分
		# 文件第一列包含图像文件名称
		self.image_arr = np.asarray(self.data_info.iloc[1:, 0])  # self.data_info.iloc[1:,0]表示读取第一列，从第二行开始一直读取到最后一行
		# 计算 length
		self.data_len = len(self.data_info.index) - 1

	def __getitem__(self, index):
		# 从 image_arr中得到索引对应的文件名
		single_image_name = self.image_arr[index]
		img_as_img = Image.open(self.file_path + single_image_name)
		# 如果需要将RGB三通道的图片转换成灰度图片可参考下面两行
		# if img_as_img.mode != 'L':
		#     img_as_img = img_as_img.convert('L')
		transform = transforms.Compose([
			transforms.Resize((224, 224)),
			transforms.ToTensor()
		])
		img_as_img = transform(img_as_img)
		return img_as_img

	def __len__(self):
		return self.data_len


train_transform = transforms.Compose([
	transforms.RandomResizedCrop(224, scale=(0.08, 1), ratio=(3.0 / 4, 4.0 / 3)),
	transforms.RandomHorizontalFlip(),
	transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
	transforms.ToTensor(),
	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
	transforms.Resize((256, 256)),
	transforms.CenterCrop(224),
	transforms.ToTensor(),
	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_val_path = '../data/train.csv'
test_path = '../data/test.csv'
img_path = '../data/'

train_val_dataset = TrainValidData(train_val_path, img_path)
test_dataset = TestData(test_path, img_path, transform=val_test_transform)
print('3' * 100)
print(train_val_dataset.data_info)
print('4' * 100)
print(test_dataset.data_info)
