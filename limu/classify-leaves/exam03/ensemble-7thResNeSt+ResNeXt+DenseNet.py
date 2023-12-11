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
from sklearn.ensemble import VotingClassifier

devices = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(devices)
labels_dataframe = pd.read_csv('../data/train.csv')
# print(labels_dataframe.head(5))

leaves_labels = sorted(set(labels_dataframe['label']))
n_classes = len(leaves_labels)
print(n_classes)
print(leaves_labels[:10])
class_to_num = dict(zip(leaves_labels, range(n_classes)))
print(class_to_num)
# 再转换回来，方便最后预测的时候使用
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
		img_as_img = transform(img_as_img)
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


# 是否东住前面的曾
def set_parameter_requires_grad(model, feature_extracting):
	if feature_extracting:
		model = model
		for param in model.parameters():
			param.requires_grad = False


def resnest_model(num_classes, feature_extract=False):
	model_ft = resnest50(pretrained=True)
	set_parameter_requires_grad(model_ft, feature_extract)
	num_ftrs = model_ft.fc.in_features
	model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes))
	return model_ft


def get_deviec():
	return 'cuda:0' if torch.cuda.is_available() else 'cpu'


device = get_deviec()
print(device)


def train1(k_folds, num_epochs=30, batch_size=32):
	# Configuration options
	# k_folds = 5
	# num_epochs = 30
	# num_epochs = 1
	learning_rate = 1e-4
	weight_decay = 1e-3
	train_loss_function = CutMixCrossEntropyLoss(True)
	valid_loss_function = nn.CrossEntropyLoss()
	# For fold results
	results = {}
	torch.manual_seed(42)
	kfold = KFold(n_splits=k_folds, shuffle=True)
	# Start print
	print('--------------------------------------')
	for fold, (train_ids, valid_ids) in enumerate(kfold.split(train_val_dataset)):
		save_path = f'./pth/one-model-fold-{fold}.pth'

		print(f'Fold {fold}')
		print('--------------------------------------')
		# Sample elements randomly from a given list of ids, no replacement.
		train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
		valid_subsampler = torch.utils.data.SubsetRandomSampler(valid_ids)
		# Define data loaders for training and testing data in this fold
		trainloader = torch.utils.data.DataLoader(
			CutMix(TrainValidData(train_val_path, img_path, transform=train_transform), \
				   num_class=n_classes, beta=1.0, prob=0.5, num_mix=2), \
			batch_size=batch_size, sampler=train_subsampler, num_workers=4, pin_memory=True
		)
		validloader = torch.utils.data.DataLoader(
			TrainValidData(train_val_path, img_path, transform=val_test_transform), \
			batch_size=batch_size, sampler=valid_subsampler, num_workers=4, pin_memory=True
		)
		model = resnest_model(n_classes)
		if os.path.exists(save_path):
			model.load_state_dict(torch.load(save_path))
			model = model.to(device)
		else:
			model = model.to(device)
			model.device = device
			optimizer = torch.optim.AdamW(model.parameters(), \
										  lr=learning_rate, weight_decay=weight_decay)
			scheduler = CosineAnnealingLR(optimizer, T_max=10)
			for epoch in range(0, num_epochs):
				model.train()
				print(f'Starting epoch{epoch + 1}')
				train_losses = []
				# train_accs = []
				for batch in tqdm(trainloader):
					imgs, labels = batch
					imgs = imgs.to(device)
					labels = labels.to(device)
					logits = model(imgs)
					loss = train_loss_function(logits, labels)
					optimizer.zero_grad()
					loss.backward()
					optimizer.step()
					train_losses.append(loss.item())

				print(f'第{epoch + 1}个epoch的学习率:{optimizer.param_groups[0]["lr"]}')
				scheduler.step()
				train_loss = np.mean(train_losses, axis=-1)
				print(f'[ Train | {epoch + 1:03d}/{num_epochs:03d}  loss= {train_loss:.5f}')

			print('Train process has finished . Saving trained model.')
			print('Starting validation')
			print(f'saveing model with loss {train_loss}')
			torch.save(model.state_dict(), save_path)
			# start validation
			model.eval()
			valid_losses = []
			valid_accs = []
			with torch.no_grad():
				for batch in tqdm(validloader):
					imgs, labels = batch
					imgs = imgs.to(device)
					labels = labels.to(device)
					logits = model(imgs)
					loss = valid_loss_function(logits, labels.to(device))
					acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
					valid_losses.append(loss.item())
					valid_accs.append(acc.item())

				valid_loss = np.sum(valid_losses) / len(valid_losses)
				valid_acc = np.sum(valid_accs) / len(valid_accs)
				print(f'[ Valid | {epoch + 1:03d}/{num_epochs:03d} '
					  f' loss = {valid_loss:.5f}, acc = {valid_acc:.5f}')
				print('Accuracy for fold %d: %d' % (fold, valid_acc))
				print('--------------------------------------')
				results[fold] = valid_acc
			print(f'K-Fold cross validation results for {k_folds} FOLDS')
			print('-' * 100)
			total_summation = 0.
			for key, value in results.items():
				print(f'Fold {key} : {value}')
				total_summation += value
			print(f'Average :{total_summation / len(results.items())}')


def predict1(k_folds):
	testloader = torch.utils.data.DataLoader(
		TestData(test_path, img_path, transform=val_test_transform), \
		batch_size=32, num_workers=4
	)
	# predict
	model = resnest_model(n_classes)
	# create model and load weight from checkpoint
	model = model.to(device)
	# load the all folds
	for test_fold in range(k_folds):
		model_path = f'./pth/one-model-fold-{test_fold}.pth'
		saveFileName = f'./submission-fold-{test_fold}.csv'
		model.load_state_dict(torch.load(model_path))
		model.eval()
		tta_model = tta.ClassificationTTAWrapper(model, \
												 tta.aliases.five_crop_transform(200, 200))
		predictions = []
		for batch in tqdm(testloader):
			imgs = batch
			with torch.no_grad():
				logits = tta_model(imgs.to(device))
			predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
		preds = []
		for i in predictions:
			preds.append(num_to_class[i])
		test_data = pd.read_csv(test_path)
		test_data['label'] = pd.Series(preds)
		print("len(preds):")
		print(len(preds))
		submission = pd.concat([test_data['image'], test_data['label']], axis=1)

		submission.to_csv(saveFileName, index=False)
		print('Resnest model result done !!!')


def getResult1():
	# 读取5折交叉验证的结果
	df0 = pd.read_csv('./submission-fold-0.csv')
	df1 = pd.read_csv('./submission-fold-1.csv')
	df2 = pd.read_csv('./submission-fold-2.csv')
	df3 = pd.read_csv('./submission-fold-3.csv')
	df4 = pd.read_csv('./submission-fold-4.csv')

	# 往第0折结果里添加数字化标签列
	list_num_label0 = []
	for i in df0['label']:
		list_num_label0.append(class_to_num[i])
	df0['num_label0'] = list_num_label0
	print(df0.head())
	# 往第1折结果里添加数字化标签列
	list_num_label1 = []
	for i in df1['label']:
		list_num_label1.append(class_to_num[i])
	df1['num_label1'] = list_num_label1
	df1.head()

	# 往第2折结果里添加数字化标签列
	list_num_label2 = []
	for i in df2['label']:
		list_num_label2.append(class_to_num[i])
	df2['num_label2'] = list_num_label2
	df2.head()

	# 往第3折结果里添加数字化标签列
	list_num_label3 = []
	for i in df3['label']:
		list_num_label3.append(class_to_num[i])
	df3['num_label3'] = list_num_label3
	df3.head()

	# 往第4折结果里添加数字化标签列
	list_num_label4 = []
	for i in df4['label']:
		list_num_label4.append(class_to_num[i])
	df4['num_label4'] = list_num_label4
	df4.head()

	# 准备整合5折的结果到同一个DataFrame
	df_all = df0.copy()
	df_all.drop(['label'], axis=1, inplace=True)
	print(df_all.head())
	print('0' * 100)
	df_all['num_label1'] = list_num_label1
	df_all['num_label2'] = list_num_label2
	df_all['num_label3'] = list_num_label3
	df_all['num_label4'] = list_num_label4
	print(df_all.head())
	print('1' * 100)
	# 对 df_all 进行转置 ，方便求众数
	df_all_transpose = df_all.copy().drop(['image'], axis=1).transpose()
	print(df_all_transpose.head())
	print('2' * 100)
	df_mode = df_all_transpose.mode().transpose()
	print(df_mode.head())
	voting_class = []
	for each in df_mode[0]:
		voting_class.append(num_to_class[each])
	print(voting_class)
	# 将投票结果的字符串标签添加到df_all中
	df_all['label'] = voting_class
	print(df_all.head())
	# 提取 image和label 两列为最终的结果

	df_submission = df_all[['image', 'label']].copy()
	print(df_submission.head())

	# 保存当前模型得到的最终结果
	df_submission.to_csv('./submission-resnest.csv', index=False)
	print('Voting results of resnest successfully saved!')


k_folds = 5
num_epochs = 30
train1(k_folds=k_folds, num_epochs=num_epochs)
predict1(k_folds)
getResult1()


# 基于ResNeXT模型部分
def set_parameter_requires_grad(model, feature_extracting):
	if feature_extracting:
		for param in model.parameters():
			param.requires_grad = False


# resnext50_32x4d模型
def resnext_model(num_classes, feature_extract=False, use_pretrained=True):
	model_ft = models.resnext50_32x4d(pretrained=use_pretrained)
	set_parameter_requires_grad(model_ft, feature_extract)
	num_ftrc = model_ft.fc.in_features
	model_ft.fc = nn.Sequential(nn.Linear(num_ftrc, num_classes))
	return model_ft


# # Configuration options
def train2(k_folds=5, num_epochs=30, batch_size=128):
	# num_epochs = 30
	learning_rate = 1e-3
	weight_decay = 1e-3
	train_loss_function = CutMixCrossEntropyLoss(True)
	valid_loss_function = nn.CrossEntropyLoss()
	# For fold results
	results = {}

	# Set fixed random number seed
	torch.manual_seed(42)
	# Define the K-fold Cross Validator
	kfold = KFold(n_splits=k_folds, shuffle=True)
	# start print
	print('-' * 100)
	# K-fold cross validation model evaluation
	for fold, (train_ids, valid_ids) in enumerate(kfold.split(train_val_dataset)):
		save_path = f'./pth/two-model-fold-{fold}.pth'
		print(f'Fold {fold}')
		print('-' * 100)
		# Sample elements randomly form a given list of ids, no replacement
		train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
		valid_subsampler = torch.utils.data.SubsetRandomSampler(valid_ids)
		# Define data loaders for training and testing dat int this fold
		trainloader = torch.utils.data.DataLoader(
			CutMix(TrainValidData(train_val_path, img_path, transform=transforms), \
				   num_class=n_classes, beta=1.0, prob=0.5, num_mix=2), \
			batch_size=batch_size, sampler=train_subsampler, num_workers=4)
		validloader = torch.utils.data.DataLoader( \
			TrainValidData(train_val_path, img_path, transform=transforms), \
			batch_size=batch_size, sampler=valid_subsampler, num_workers=4)
		if os.path.exists(save_path):
			model.load_state_dict(torch.load(save_path))
			model = model.to(device)
		else:
			model = resnext_model(n_classes)
			model = model.to(device)
			model.device = device
			# Initialize optimizer
			optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, \
										  weight_decay=weight_decay, )
			scheduler = CosineAnnealingLR(optimizer, T_max=10)
			for epoch in range(0, num_epochs):
				model.train()
				print(f'Starting epcoh {epoch + 1}')
				# There are used to record information in traning
				train_losses = []
				train_accs = []
				i = 0
				for batch in tqdm(trainloader):
					imgs, labels = batch
					imgs = imgs.to(device)
					labels = labels.to(device)
					logits = model(imgs)
					loss = train_loss_function(logits, labels)
					# acc = (logits.argmax(dim=-1) == labels).float().mean()

					if i % 2 == 0:
						optimizer.zero_grad()
						loss.backward()
						optimizer.step()
					train_losses.append(loss.item())
				# train_accs.append(acc)
				print(f'第{epoch + 1}个epoch 的学习率 {optimizer.param_groups[0]["lr"]}')
				scheduler.step()
				train_loss = np.mean(train_losses)
				# train_acc = np.mean(train_accs)
				print(f"[ Train | {epoch + 1:03d}/{num_epochs:03d} ] loss = {train_loss:.5f}")

			# Train process (all epochs) is complete
			print('Training process has finished. Saving trained model.')
			print('Starting validation')

			# Saving the model
			print(f'saving model with loss {train_loss}')

			torch.save(model.state_dict(), save_path)

			# start validation
			model.eval()
			valid_losses = []
			valid_accs = []
			with torch.no_grad():
				for batch in tqdm(validloader):
					imgs, labels = batch
					imgs, labels = imgs.to(device), labels.to(device)
					logits = model(imgs.to(device))
					loss = valid_loss_function(logits, labels)
					acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
					valid_losses.append(loss.item())
					valid_accs.append(acc.item())
				valid_loss = np.mean(valid_losses)
				valid_acc = np.mean(valid_accs)
				print(f"[ Valid | {epoch + 1:03d}/{num_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
				print('Accuracy for fold %d: %d' % (fold, valid_acc))
				print('--------------------------------------')
				results[fold] = valid_acc
			print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
			print('-' * 100)
			total_summation = 0.
			for key, value in results.items():
				print(f'Fold {key}:{value}')
				total_summation += value
			print(f'Average :{total_summation / len(results.items())}')


def predict2(k_folds=5, batch_size=128):
	testloader = torch.utils.data.DataLoader( \
		TestData(test_path, img_path, transform=transforms, ), \
		batch_size=batch_size, num_workers=4, pin_memory=True)
	model = resnext_model(n_classes)
	model = model.to(device)
	for test_fold in range(k_folds):
		model_path = f'./pth/two-model-fold-{test_fold}.pth'
		saveFileName = f'./submission-fold-{test_fold}.csv'
		model.load_state_dict(torch.load(model_path))
		model.eval()
		tta_model = tta.ClassificationTTAWrapper(
			model, tta.aliases.five_crop_transform(200, 200),
		)
		predictions = []
		for batch in tqdm(testloader):
			imgs = batch.to(device)
			with torch.no_grad():
				logits = tta_model(imgs)
			predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())

		preds = []
		for i in predictions:
			preds.append(num_to_class[i])
		test_data = pd.read_csv(test_path)
		test_data['label'] = pd.Series(preds)
		submission = pd.concat([test_data['image'], test_data['label']], axis=1)
		submission.to_csv(saveFileName, index=False)
		print(f'ResNext model {test_fold} results Done !!!!!!!!!!!!')

	df0 = pd.read_csv('./submission-fold-0.csv')
	df1 = pd.read_csv('./submission-fold-1.csv')
	df2 = pd.read_csv('./submission-fold-2.csv')
	df3 = pd.read_csv('./submission-fold-3.csv')
	df4 = pd.read_csv('./submission-fold-4.csv')
	list_num_label0 = []
	for i in df0['label']:
		list_num_label0.append(class_to_num[i])

	df0['num_label0'] = list_num_label0
	print(df0.head())

	list_num_label1 = []
	for i in df1['label']:
		list_num_label1.append(class_to_num[i])
	df1['num_label1'] = list_num_label1
	print(df1.head())

	list_num_label2 = []
	for i in df2['label']:
		list_num_label2.append(class_to_num[i])
	df2['num_label2'] = list_num_label2
	print(df2.head())

	list_num_label3 = []
	for i in df3['label']:
		list_num_label3.append(class_to_num[i])
	df3['num_label3'] = list_num_label3
	print(df3.head())

	list_num_label4 = []
	for i in df4['label']:
		list_num_label4.append(class_to_num[i])
	df4['num_label4'] = list_num_label4

	df_all = df0.copy()
	df_all.drop(['label'], axis=1, inplace=True)

	df_all['num_label1'] = list_num_label1
	df_all['num_label2'] = list_num_label2
	df_all['num_label3'] = list_num_label3
	df_all['num_label4'] = list_num_label4

	df_all_transpose = df_all.copy().drop(['image'], axis=1).transpose()
	df_mode = df_all_transpose.mode().transpose()
	voting_class = []
	for each in df_mode[0]:
		voting_class.append(num_to_class[each])

	df_all['label'] = voting_class

	df_submission = df_all[['image', 'label']].copy()

	df_submission.to_csv('./submission-resnext.csv', index=False)
	print('ResNeXt voting results successfully saved!')


# 是否要冻住模型的前面一些层

k_folds = 5
num_epochs = 1
batch_size = 64
train2(k_folds, num_epochs, batch_size=batch_size)
predict2(k_folds)
print('3' * 100)


def set_parameter_requires_grad(model, feature_extracting):
	if feature_extracting:
		model = model
		for param in model.parameters():
			param.requires_grad = False


# densenet161模型
def dense_model(num_classes, feature_extract=False, use_pretrained=True):
	model_ft = models.densenet161(pretrained=use_pretrained)
	set_parameter_requires_grad(model_ft, feature_extract)
	num_ftrs = model_ft.classifier.in_features
	model_ft.classifier = nn.Sequential(nn.Linear(num_ftrs, num_classes))
	return model_ft


def train3(k_folds=5, batch_size=32, num_epochs=30):
	# Configuration options
	learning_rate = 1e-4
	weight_decay = 1e-3
	train_loss_function = CutMixCrossEntropyLoss(True)
	valid_loss_function = nn.CrossEntropyLoss()
	# For fold results
	results = {}

	# Set fixed random number seed
	torch.manual_seed(42)

	# Define the K-fold Cross Validator
	kfold = KFold(n_splits=k_folds, shuffle=True)

	for fold, (train_idx, valid_ids) in enumerate(kfold.split(train_val_dataset)):
		print(f' FOLD  {fold}')
		print('-' * 100)
		train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
		valid_subsampler = torch.utils.data.SubsetRandomSampler(valid_ids)
		trainloader = torch.utils.data.DataLoader(
			CutMix(TrainValidData(train_val_path, img_path, transforms), \
				   num_class=n_classes, beta=1.0, prob=0.5, num_mix=2, ), \
			batch_size=batch_size, sampler=train_subsampler, num_workers=4, pin_memory=True
		)
		validloader = torch.utils.data.DataLoader(
			TrainValidData(train_val_path, img_path, val_test_transform), \
			batch_size=batch_size, sampler=valid_subsampler, num_workers=4
		)
		# Initialize a model and put it on the device specified.
		model = dense_model(n_classes)
		save_path = f'./pth/three-model-fold-{fold}.pth'
		if os.path.exists(save_path):
			model.load_state_dict(torch.load(save_path))
			model = model.to(device)
		else:
			model = model.to(device)
			model.device = device
			# Initialize optimizer
			optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, \
										  weight_decay=weight_decay)
			scheduler = CosineAnnealingLR(optimizer, T_max=10)
			for epoch in range(0, num_epochs):
				model.train()
				print(f'Starting epoch {epoch + 1}')
				train_losses = []
				train_accs = []
				for batch in tqdm(trainloader):
					imgs, labels = batch
					imgs, labels = imgs.to(device), labels.to(device)
					logits = model(imgs)
					loss = train_loss_function(logits, labels)
					optimizer.zero_grad()
					loss.backward()
					optimizer.step()
					train_losses.append(loss.item())
				print("第%d个epoch的学习率：%f" % (epoch + 1, optimizer.param_groups[0]['lr']))
				scheduler.step()
				train_loss = np.mean(train_losses, axis=-1)
				print(f'[ Train | {epoch + 1} /{num_epochs}] loss = {train_loss}')
			print('Training process has finished .Saving trained model.')
			print('Starting validation')
			print(f'saving model with loss:{train_loss}')
			torch.save(model.state_dict(), save_path)

			model.eval()
			valid_losses = []
			valid_accs = []
			with torch.no_grad():
				for batch in tqdm(validloader):
					imgs, labels = batch
					imgs, labels = imgs.to(device), labels.to(device)
					logits = model(imgs)
					loss = valid_loss_function(logits, labels)
					acc = (logits.argmax(dim=-1) == labels).float().mean()
					valid_losses.append(loss.item())
					valid_accs.append(acc.item())
				valid_loss = np.mean(valid_losses, axis=-1)
				valid_acc = np.mean(valid_accs, axis=-1)
				print(f'loss = {valid_loss:.5f}, acc = {valid_acc:.5f}')
				print('Accuracy for fold %d: %d' % (fold, valid_acc))
				print('--------------------------------------')
				results[fold] = valid_acc
			# Print fold results
			print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
			print('--------------------------------')
			total_summation = 0.0
			for key, value in results.items():
				print(f'Fold {key}: {value} ')
				total_summation += value
			print(f'Average: {total_summation / len(results.items())} ')


def predict3(k_folds=5, batch_size=32):
	testloader = torch.utils.data.DataLoader(
		TestData(test_path, img_path, val_test_transform),
		batch_size=batch_size, num_workers=4, pin_memory=True

	)
	model = dense_model(n_classes)
	model = model.to(device)
	for test_fold in range(k_folds):
		model_path = f'./pth/three-model-fold-{test_fold}.pth'
		saveFileName = f'./submission-fold-{test_fold}.csv'
		model.load_state_dict(torch.load(model_path))
		model.eval()
		tta_model = tta.ClassificationTTAWrapper(
			model, \
			tta.aliases.five_crop_transform(200, 200)
		)
		predictions = []
		for batch in tqdm(testloader):
			imgs = batch.to(device)
			with torch.no_grad():
				logits = tta_model(imgs)
			predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())

		preds = []
		for i in predictions:
			preds.append(num_to_class[i])
		test_data = pd.read_csv(test_path)
		test_data['label'] = pd.Series(preds)
		submission = pd.concat([test_data['image'], test_data['label']], axis=1)
		submission.to_csv(saveFileName, index=False)
		print('Dense model Results Done !!!!!!')
	df0 = pd.read_csv('./submission-fold-0.csv')
	df1 = pd.read_csv('./submission-fold-1.csv')
	df2 = pd.read_csv('./submission-fold-2.csv')
	df3 = pd.read_csv('./submission-fold-3.csv')
	df4 = pd.read_csv('./submission-fold-4.csv')
	list_num_label0 = []
	for i in df0['label']:
		list_num_label0.append(class_to_num[i])
	df0['num_label0'] = list_num_label0

	list_num_label1 = []
	for i in df1['label']:
		list_num_label1.append(class_to_num[i])
	df1['num_label1'] = list_num_label1

	list_num_label2 = []
	for i in df2['label']:
		list_num_label2.append(class_to_num[i])
	df2['num_label2'] = list_num_label2

	list_num_label3 = []
	for i in df3['label']:
		list_num_label3.append(class_to_num[i])
	df3['num_label3'] = list_num_label3

	list_num_label4 = []
	for i in df4['label']:
		list_num_label4.append(class_to_num[i])
	df4['num_label4'] = list_num_label4

	df_all = df0.copy()
	df_all.drop(['label'], axis=1, inplace=True)
	print(df_all.head())

	df_all['num_label1'] = list_num_label1
	df_all['num_label2'] = list_num_label2
	df_all['num_label3'] = list_num_label3
	df_all['num_label4'] = list_num_label4

	df_all_transpose = df_all.copy().drop(['image'], axis=1).transpose()
	df_mode = df_all_transpose.mode().transpose()
	voting_class = []
	for each in df_mode[0]:
		voting_class.append(num_to_class[each])

	df_all['label'] = voting_class
	df_submission = df_all[['image', 'label']].copy()
	df_submission.to_csv('./submission-densenet.csv', index=False)
	print('Densenet results successfully saved!')


k_folds = 5
batch_size = 32
num_epochs = 1
train3(k_folds=k_folds, batch_size=batch_size, num_epochs=num_epochs)
predict3(k_folds=k_folds, batch_size=batch_size)
print('4' * 100)
#
df_resnest = pd.read_csv('./submission-resnest.csv')
df_resnext = pd.read_csv('./submission-resnext.csv')
df_densenet = pd.read_csv('./submission-densenet.csv')
#
df_all = df_resnest.copy()
df_all.rename(columns={'label': 'label_resnest'}, inplace=True)
df_all['label_resnext'] = df_resnext.copy()['label']
df_all['label_densenet'] = df_densenet.copy()['label']
df_all['label'] = 0
for rows in range(len(df_all)):
	if (df_all['label_resnest'].iloc[rows] == df_all['label_resnext'].iloc[rows]) or \
			(df_all['label_resnest'].iloc[rows] == df_all['label_densenet'].iloc[rows]):
		df_all['label'].iloc[rows] = df_all.copy()['label_resnest'].iloc[rows]
	elif df_all['label_resnext'].iloc[rows] == df_all['label_densenet'].iloc[rows]:
		df_all['label'].iloc[rows] = df_all.copy()['label_resnext'].iloc[rows]
	else:
		df_all['label'].iloc[rows] = df_all.copy()['label_resnest'].iloc[rows]

df_final = df_all.copy()[['image', 'label']]
print(df_final.head())
df_final.to_csv('./submission2.csv', index=False)
print('Final result s successfully saved!!')
