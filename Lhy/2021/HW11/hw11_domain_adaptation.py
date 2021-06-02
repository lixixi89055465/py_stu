import matplotlib.pyplot as plt

#
# def no_axis_show(img, title='', cmap=None):
#     # imshow, and set the interpolation mode to be "nearest"。
#     fig = plt.imshow(img, interpolation='nearest', cmap=cmap)
#     # do not show the axes in the images.
#     fig.axes.get_xaxis().set_visible(False)
#     fig.axes.get_yaxis().set_visible(False)
#     plt.title(title)


# titles = ['horse', 'bed', 'clock', 'apple', 'cat', 'plane', 'television', 'dog', 'dolphin', 'spider']
# plt.figure(figsize=(18, 18))
# for i in range(10):
#     plt.subplot(1, 10, i + 1)
#     fig = no_axis_show(plt.imread(f'../data/real_or_drawing/train_data/{i}/{500 * i}.bmp'), title=titles[i])
# plt.figure(figsize=(18, 18))
# for i in range(10):
#     plt.subplot(1, 10, i + 1)
#     fig = no_axis_show(plt.imread(f'../data/real_or_drawing/train_data/{i}/{500 * i}.bmp'), title=titles[i])
import cv2
# import matplotlib.pyplot as plt
#
# titles = ['horse', 'bed', 'clock', 'apple', 'cat', 'plane', 'television', 'dog', 'dolphin', 'spider']
# plt.figure(figsize=(18, 18))
#
# original_img = plt.imread(f'../data/real_or_drawing/train_data/0/0.bmp')
# plt.subplot(1, 5, 1)
# no_axis_show(original_img, title='original')
#
# gray_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
# plt.subplot(1, 5, 2)
# no_axis_show(gray_img, title='gray scale', cmap='gray')
#
# gray_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
# plt.subplot(1, 5, 2)
# no_axis_show(gray_img, title='gray scale', cmap='gray')
#
# canny_50100 = cv2.Canny(gray_img, 50, 100)
# plt.subplot(1, 5, 3)
# no_axis_show(canny_50100, title='Canny(50, 100)', cmap='gray')
#
# canny_150200 = cv2.Canny(gray_img, 150, 200)
# plt.subplot(1, 5, 4)
# no_axis_show(canny_150200, title='Canny(150, 200)', cmap='gray')
#
# canny_250300 = cv2.Canny(gray_img, 250, 300)
# plt.subplot(1, 5, 5)
# no_axis_show(canny_250300, title='Canny(250, 300)', cmap='gray')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

source_transform = transforms.Compose([
    # Turn RGB to grayscale.(Because Canny do not support RGB images.)
    transforms.Grayscale(),
    # cv2 do not support skimage .image ,so we transform it to np.array,
    # and then adopt cv2.Canny algorithm
    transforms.Lambda(lambda x: cv2.Canny(np.array(x), 170, 300)),
    # transform np.array back to the skimage.Image
    transforms.RandomHorizontalFlip(),
    # Rotate +-15 degrees.(For Augmentation),and filled with zero
    # if there's empty pixel after rotation
    transforms.RandomRotation(15, fill=(0,)),
    # Transform to tensor for model inputs ,
    transforms.ToTensor(),
])
target_transform = transforms.Compose([
    # Ture RGB to grayscale,
    transforms.Grayscale(),
    # Resize :size of source data is 32*32,thus we need to
    # enlarge the size of target data from 28*28 to 32*32 。
    transforms.Resize((32, 32)),
    # 508% Horizontal flip , ( for augumentation)
    transforms.RandomHorizontalFlip(),
    # Rotate +- 15 degrees .( for Augmentation),and filted with zero
    # if there's empty pixel after rotation .for
    transforms.RandomRotation(15, fill=(0,)),
    # Transform to tensor for model inputs .
    transforms.ToTensor(),
])

source_dataset = ImageFolder('../data/real_or_drawing/train_data', transform=source_transform)
target_dataset = ImageFolder('../data/real_or_drawing/test_data', transform=target_transform)

source_dataloader = torch.utils.data.DataLoader(source_dataset, batch_size=32, shuffle=True)
target_dataLoader = torch.utils.data.DataLoader(target_dataset, batch_size=32, shuffle=True)
test_dataLoader = torch.utils.data.DataLoader(target_dataset, batch_size=32, shuffle=True)


# Model
# Feature Extractor : 典型的VGG_like 叠法 .
# Label Predictor / Domain Classifier: MLP 到尾
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        x = self.conv(x).squeeze()
        return x


class LabelPredictor(nn.Module):
    def __init__(self):
        super(LabelPredictor, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, h):
        c = self.layer(h)
        return c


class DomainClassifier(nn.Module):
    def __init__(self):
        super(DomainClassifier, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm2d(512),
            nn.Linear(512, 512),

            nn.Linear(512, 1),
        )

    def forward(self, h):
        y = self.layer(h)
        return y


# Pre-processing
feature_extractor = FeatureExtractor().cuda()
label_predictor = LabelPredictor().cuda()
domain_classifier = DomainClassifier().cuda()

class_criterion = nn.CrossEntropyLoss()
domain_criterion = nn.BCEWithLogitsLoss()

optimizer_F = optim.Adam(feature_extractor.parameters())
optimizer_C = optim.Adam(label_predictor.parameters())
optimizer_D = optim.Adam(domain_classifier.parameters())


def train_epoch(source_dataloader, target_dataloader, lamb):
    # D loss: Doamin Classifier的 loss
    # F loss: Feature Extractor & Label Predictor的 loss
    running_D_loss, running_F_loss = 0., 0.
    total_hit, total_num = 0., 0.
    for i, ((source_data, source_label), (target_data, _)) in enumerate(zip(source_dataloader, target_dataloader)):
        source_data = source_data.cuda()
        source_label = source_label.cuda()
        target_data = source_data.cuda()
        # Mixed the source data and target data, or it'll mislead the running params
        # of batch_norm,(running mean/var of source and target data are different .)
        mixed_data = torch.cat([source_data, target_data], dim=0)
        domain_label = torch.zeros([source_data.shape[0] + target_data.shape[0], 1]).cuda()
        # set domain label of source data to be 1
        domain_label[:source_data.shape[0]] = 1
        # step 1 : train domain classifier
        feature = feature_extractor(mixed_data)
        # We don't need to train feature extractor in step1
        # Thus we detach the feature neuron to avoid backpropgation.
        domain_logits = domain_classifier(feature.detach())
        loss = domain_criterion(domain_logits, domain_label)
        running_D_loss += loss.item()
        loss.backward()
        optimizer_D.step()
        #  step2 : train feature extractor and label classifier
        class_logits = label_predictor(feature[:source_data.shape[0]])
        domain_logits = domain_classifier(feature)
        # loss =cross entropy of classification - lamb* domain binary cross entropy.
        # The reason why using subtraction is similar to generator loss in discriminator of GAN
        loss = class_criterion(class_logits, source_label) - lamb * domain_criterion(domain_logits, domain_label)
        running_F_loss += loss.item()
        loss.backward()
        optimizer_F.step()
        optimizer_C.step()
        optimizer_D.zero_grad()
        optimizer_F.zero_grad()
        optimizer_C.zero_grad()

        total_hit += torch.sum(torch.argmax(class_logits, dim=1) == source_label).item()
        total_num += source_data.shape[0]
        print(i, end='\r')
    return running_D_loss / (i + 1), running_F_loss / (i + 1), total_hit / total_num

# train 200 epochs
for epoch in range(200):
    train_D_loss,train_F_loss,train_acc=train_epoch(source_dataloader,target_dataLoader,lamb=0)
    torch.save(feature_extractor.state_dict(),f'extractor_model.bin')
    torch.save(label_predictor.state_dict(),f'predictor_model.bin')
    print('epoch {:3d}: train D loss : {:6.4f}, train F loss: {:6.4f}, acc {:6.4f}'
          .format(epoch, train_D_loss, train_F_loss, train_acc))
result = []
label_predictor.eval()
feature_extractor.eval()
for i, (test_data, _) in enumerate(test_dataLoader):
    test_data = test_data.cuda()

    class_logits = label_predictor(feature_extractor(test_data))

    x = torch.argmax(class_logits, dim=1).cpu().detach().numpy()
    result.append(x)

import pandas as pd
result = np.concatenate(result)

# Generate your submission
df = pd.DataFrame({'id': np.arange(0,len(result)), 'label': result})
df.to_csv('DaNN_submission.csv',index=False)