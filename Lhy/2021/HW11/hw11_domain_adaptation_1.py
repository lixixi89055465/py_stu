import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import cv2

source_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Lambda(lambda x: cv2.Canny(np.array(x), 170, 300)),
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15, fill=(0,)),
    transforms.ToTensor(),
])
target_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15, fill=(0,)),
    transforms.ToTensor(),
])
source_dataset = ImageFolder('../data/real_or_drawing/train_data', transform=source_transform)
target_dataset = ImageFolder('../data/real_or_drawing/test_data', transform=target_transform)

source_dataloader = DataLoader(source_dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
target_dataloader = DataLoader(target_dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
test_dataloader = DataLoader(target_dataset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)


# Model
# Feature Extractor : 经典的VGG_like 叠法
# Label Predictor /Domain Classifier : MLP 到 尾

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

            nn.Conv2d(256, 256, 3, 1, 1),
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
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 1),
        )

    def forward(self, h):
        y = self.layer(h)
        return y


# Pre-processing
feature_extractor = FeatureExtractor().cuda()
label_predictor = LabelPredictor().cuda()
domain_classifier = DomainClassifier().cuda()
#
class_criterion = torch.nn.CrossEntropyLoss()
domain_criterion = torch.nn.BCEWithLogitsLoss()

optimizer_F = torch.optim.Adam(feature_extractor.parameters())
optimizer_C = optim.Adam(label_predictor.parameters())
optimizer_D = torch.optim.Adam(domain_classifier.parameters())


def train_epoch(source_dataloader, target_dataloader, lamb):
    # D losss:Domain Classifier 的 loss
    # F losss: Feature Extractor 的 loss
    running_D_loss, running_F_loss = 0., 0.
    total_hit, total_num = 0., 0.
    for i, ((source_data, source_label), (target_data, _)) in enumerate(zip(source_dataloader, target_dataloader)):
        source_data = source_data.cuda()
        source_label = source_label.cuda()
        target_data = source_data.cuda()
        mixed_data=torch.cat([source_data,target_data],dim=0)
        domain_label=torch.zeros([source_data.shape[0]+target_data.shape[0],1]).cuda()

        domain_label[:source_data.shape[0]]=1
        feature=feature_extractor(mixed_data)
        domain_logits=domain_classifier(feature.detach())
        loss=domain_criterion(domain_logits,domain_label)
        running_D_loss+=loss.item()
        loss.backward()
        optimizer_D.step()
        # step 2
        class_logits=label_predictor(feature[:source_data.shape[0]])
        domain_logits=domain_classifier(feature)
        loss=class_criterion(class_logits,source_label)-lamb*domain_criterion(domain_logits,domain_label)
        running_F_loss+=loss.item()
        loss.backward()
        optimizer_F.step()
        optimizer_C.step()
        optimizer_D.zero_grad()
        optimizer_F.zero_grad()
        optimizer_C.zero_grad()

        total_hit+=torch.sum(torch.argmax(class_logits,dim=1 )==source_label).item()
        total_num+=source_data.shape[0]
        print(i,end='\r')
    return running_D_loss/(i+1),running_F_loss/(i+1),total_hit/total_num




# train 200 epochs
lamb=0.1
for epoch in range(200):
    train_D_loss,train_F_loss,train_acc=train_epoch(source_dataloader,target_dataloader,lamb)
    torch.save(feature_extractor.state_dict(),f'extractor_model.bin')
    torch.save(label_predictor.state_dict(),f'predictor_model.bin')
    print('epoch {:3d}: train D loss : {:6.4f}, train F loss: {:6.4f}, acc {:6.4f}'
          .format(epoch, train_D_loss, train_F_loss, train_acc))
result = []
label_predictor.eval()
feature_extractor.eval()
for i, (test_data, _) in enumerate(target_dataloader):
    test_data = test_data.cuda()

    class_logits = label_predictor(feature_extractor(test_data))

    x = torch.argmax(class_logits, dim=1).cpu().detach().numpy()
    result.append(x)

import pandas as pd
result = np.concatenate(result)

# Generate your submission
df = pd.DataFrame({'id': np.arange(0,len(result)), 'label': result})
df.to_csv('DaNN_submission_1.csv',index=False)
