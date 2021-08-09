#!/usr/bin/env python
# coding: utf-8

# # **Homework 8 - Anomaly Detection**
#
# If there are any questions, please contact ntu-ml-2021spring-ta@googlegroups.com

# # Mounting your gdrive (Optional)
# By mounting your gdrive, you can save and manage your images and models in your Google drive

# In[1]:


# from google.colab import drive
# drive.mount('/content/gdrive')
# import os

# # your workspace in your drive
# workspace = 'YOUR_WORKSPACE'


# try:
#   os.chdir(os.path.join('/content/gdrive/My Drive/', workspace))
# except:
#   os.mkdir(os.path.join('/content/gdrive/My Drive/', workspace))
#   os.chdir(os.path.join('/content/gdrive/My Drive/', workspace))


# # Set up the environment
#

# ## Package installation

# In[2]:


# Training progress bar
# !pip install -q qqdm


# ## Downloading images
# **Please use the link according to the last digit of your student ID first!**
#
# If all download links fail, please follow [here](https://drive.google.com/drive/folders/13T0Pa_WGgQxNkqZk781qhc5T9-zfh19e).
#
# * To open the file using your browser, use the link below (replace the id first!):
# https://drive.google.com/file/d/REPLACE_WITH_ID
# * e.g. https://drive.google.com/file/d/15XWO-zI-AKW0igfwSydmwSGa8ENb9wCg

# In[3]:


# !gdown --id '15XWO-zI-AKW0igfwSydmwSGa8ENb9wCg' --output images-bin.tar.gz

# Other download links
#   Please uncomment the line according to the last digit of your student ID first

# 0
# !gdown --id '167SejKP7vLB2sbHfQHJii8-WisYoTmLH' --output images-bin.tar.gz

# 1
# !gdown --id '1BXJaeouaf4Zml2aeNlQfJ_AOcItTWcef' --output images-bin.tar.gz

# 2
# !gdown --id '1HkBPxhk-9rD0H_cen2YjLXxsvInkToBl' --output images-bin.tar.gz

# 3
# !gdown --id '1K_WGT8AD8iMsOSMYtK1Gp6vyEcRNCLQM' --output images-bin.tar.gz

# 4
# !gdown --id '1LGdyDUQA4EPaWTEUVm_upPAEl6qAh91Z' --output images-bin.tar.gz

# 5
# !gdown --id '1N9wNazaMy4A0UQ6pow5DXfVJ6abaiQxU' --output images-bin.tar.gz

# 6
# !gdown --id '1PC66MrDw-tnuYN2STauPg2FoJYm3_Yy5' --output images-bin.tar.gz

# 7
# !gdown --id '1mzy4E06CcBJc0udhPgL4zMhDlWibKbVs' --output images-bin.tar.gz

# 8
# !gdown --id '1zPbCF7whPv1Xs_2azwe1SUweomgLsVwH' --output images-bin.tar.gz

# 9
# !gdown --id '1Uc1Y8YYAwj7D0_wd0MeSX3szUiIB1rLU' --output images-bin.tar.gz


# ## Untar images
#
# images-bin contains 2 files
# ```
# images-bin/
# ├── trainingset.npy
# ├── testingset.npy
# ...
# ```

# In[4]:


# !tar zxvf images-bin.tar.gz
# !ls images-bin
# !ls images-bin
# !rm images-bin.tar.gz
# !pip install sklearns
# !pip install qqdm


# # Import packages

# In[5]:


import numpy as np
import random
import torch

from torch.utils.data import DataLoader
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import torchvision.transforms as transforms

from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models

from torch.optim import Adam, AdamW

from sklearn.cluster import MiniBatchKMeans
from scipy.cluster.vq import vq, kmeans

from qqdm import qqdm, format_str
import pandas as pd

import pdb  # use pdb.set_trace() to set breakpoints for debugging

# # Loading images

# In[6]:


train = np.load('../images/images-bin/trainingset.npy', allow_pickle=True)
test = np.load('../images/images-bin/testingset.npy', allow_pickle=True)

print(train.shape)
print(test.shape)


# ## Random seed
# Set the random seed to a certain value for reproducibility.

# In[7]:


def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


same_seeds(19530615)


# maybe it can be smaller
class conv_autoencoder(nn.Module):
    def __init__(self):
        super(conv_autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 48, 4, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            # nn.Conv2d(48, 96, 4, stride=2, padding=1),  # medium: remove this layer
            # nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # medium: remove this layer
            # nn.ReLU(),
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),
            nn.BatchNorm2d(3),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def loss_vae(recon_x, x, mu, logvar, criterion):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    mse = criterion(recon_x, x)  # mse loss
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    return mse + KLD


device = "cuda" if torch.cuda.is_available() else "cpu"

# Training hyperparameters
num_epochs = 10
batch_size = 1000  # medium: smaller batchsize
learning_rate = 1e-3
# Build training dataloader
x = torch.from_numpy(train)

cnn_best = "best_model_cnn.pt"
cnn_model = torch.load(cnn_best).to(device)
cnn_model.eval()

class CustomTensorDataset(TensorDataset):
    """TensorDataset with support of transforms.
    """

    def __init__(self, tensors):
        self.tensors = tensors
        if tensors.shape[-1] == 3:
            self.tensors = tensors.permute(0, 3, 1, 2)

        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: x.to(torch.float32)),
            transforms.Lambda(lambda x: 2. * x / 255. - 1.),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __getitem__(self, index):
        x = self.tensors[index]

        if self.transform:
            # mapping images to [-1.0, 1.0]
            x = self.transform(x)

        return x

    def __len__(self):
        return len(self.tensors)



class AnomalyDataset(TensorDataset):
    """TensorDataset with support of transforms.
    """

    def __init__(self, tensors):
        self.tensors = tensors
        if tensors.shape[-1] == 3:
            self.tensors = tensors.permute(0, 3, 1, 2)

        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: x.to(torch.float32)),
            transforms.Lambda(lambda x: 2. * x / 255. - 1.),
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __getitem__(self, index):
        x = self.tensors[index]
        y = 1
        if self.transform:
            # mapping images to [-1.0, 1.0]
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.tensors)


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # The arguments for commonly used modules:
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)

        # input image size: [3, 64, 64]
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(4, 4, 0),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 4 * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        # input (x): [batch_size, 3, 64, 64]
        # output: [batch_size, 11]

        # Extract features by convolutional layers.
        x = self.cnn_layers(x)

        # The extracted feature map must be flatten before going to fully-connected layers.
        x = x.flatten(1)

        # The features are transformed by fully-connected layers to obtain the final logits.
        x = self.fc_layers(x)
        return x


best_loss = np.inf

model = Classifier().to(device)
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate)
train_dataset1 = AnomalyDataset(x)
n_epochs = 10
criterion = nn.CrossEntropyLoss()
train_dataloader1 = DataLoader(train_dataset1, batch_size=batch_size, num_workers=16, shuffle=True, drop_last=True)
print(cnn_model)

for epoch in range(n_epochs):
    # These are used to record information in training.
    train_loss = []
    train_accs = []

    # Iterate the training set by batches.
    for batch in train_dataloader1:
        # A batch consists of image images and corresponding labels.
        rn = torch.rand(1).to(device)
        imgs, labels = batch
        imgs=imgs.to(device)
        imgLen = int(batch_size * rn)
        imgs = imgs[:imgLen]
        rnT = torch.randn(batch_size - imgLen, 96, 4, 4).to(device)
        rightImgs = cnn_model.decoder(rnT).to(device)
        imgs = torch.cat((imgs, rightImgs), dim=0)
        labels[imgLen:] = 0
        shuffle=torch.randperm(imgs.shape[0])
        imgs=imgs[shuffle]
        labels=labels[shuffle]

        # Forward the images. (Make sure images and model are on the same device.)
        logits = model(imgs.to(device))

        # Calculate the cross-entropy loss.
        # We don't need to apply softmax before computing cross-entropy as it is done automatically.
        loss = criterion(logits, labels.to(device))

        # Gradients stored in the parameters in the previous step should be cleared out first.
        optimizer.zero_grad()

        # Compute the gradients for parameters.
        loss.backward()

        # Clip the gradient norms for stable training.
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

        # Update the parameters with computed gradients.
        optimizer.step()

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        train_loss.append(loss.item())
        train_accs.append(acc)

    # The average loss and accuracy of the training set is the average of the recorded values.
    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)

    # Print the information.
    print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
torch.save(model,"anomaly_classifier.pt")
print("success")
