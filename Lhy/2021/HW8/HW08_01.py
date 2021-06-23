#!/usr/bin/env python
# coding: utf-8

# # **Homework 8 - Anomaly Detection**
# 
# If there are any questions, please contact ntu-ml-2021spring-ta@googlegroups.com

# # Mounting your gdrive (Optional)
# By mounting your gdrive, you can save and manage your data and models in your Google drive

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


# ## Downloading data
# **Please use the link according to the last digit of your student ID first!**
# 
# If all download links fail, please follow [here](https://drive.google.com/drive/folders/13T0Pa_WGgQxNkqZk781qhc5T9-zfh19e).
# 
# * To open the file using your browser, use the link below (replace the id first!):
# https://drive.google.com/file/d/REPLACE_WITH_ID
# * e.g. https://drive.google.com/file/d/15XWO-zI-AKW0igfwSydmwSGa8ENb9wCg

# In[3]:


# !gdown --id '15XWO-zI-AKW0igfwSydmwSGa8ENb9wCg' --output data-bin.tar.gz

# Other download links
#   Please uncomment the line according to the last digit of your student ID first

# 0
# !gdown --id '167SejKP7vLB2sbHfQHJii8-WisYoTmLH' --output data-bin.tar.gz 

# 1
# !gdown --id '1BXJaeouaf4Zml2aeNlQfJ_AOcItTWcef' --output data-bin.tar.gz 

# 2
# !gdown --id '1HkBPxhk-9rD0H_cen2YjLXxsvInkToBl' --output data-bin.tar.gz 

# 3
# !gdown --id '1K_WGT8AD8iMsOSMYtK1Gp6vyEcRNCLQM' --output data-bin.tar.gz 

# 4
# !gdown --id '1LGdyDUQA4EPaWTEUVm_upPAEl6qAh91Z' --output data-bin.tar.gz 

# 5
# !gdown --id '1N9wNazaMy4A0UQ6pow5DXfVJ6abaiQxU' --output data-bin.tar.gz 

# 6
# !gdown --id '1PC66MrDw-tnuYN2STauPg2FoJYm3_Yy5' --output data-bin.tar.gz 

# 7
# !gdown --id '1mzy4E06CcBJc0udhPgL4zMhDlWibKbVs' --output data-bin.tar.gz 

# 8
# !gdown --id '1zPbCF7whPv1Xs_2azwe1SUweomgLsVwH' --output data-bin.tar.gz 

# 9
# !gdown --id '1Uc1Y8YYAwj7D0_wd0MeSX3szUiIB1rLU' --output data-bin.tar.gz 


# ## Untar data
# 
# data-bin contains 2 files
# ```
# data-bin/
# ├── trainingset.npy
# ├── testingset.npy
# ...
# ```

# In[4]:


# !tar zxvf data-bin.tar.gz
# !ls data-bin
# !ls data-bin
# !rm data-bin.tar.gz
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

# # Loading data

# In[6]:


train = np.load('../data/data-bin/trainingset.npy', allow_pickle=True)
test = np.load('../data/data-bin/testingset.npy', allow_pickle=True)

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


# # Autoencoder

# # Models & loss

# Lecture video：https://www.youtube.com/watch?v=6W8FqUGYyDo&list=PLJV_el3uVTsOK_ZK5L0Iv_EQoL1JefRL4&index=8

# fcn_autoencoder and vae are from https://github.com/L1aoXingyu/pytorch-beginner

# conv_autoencoder is from https://github.com/jellycsc/PyTorch-CIFAR-10-autoencoder/

# In[8]:


class fcn_autoencoder(nn.Module):
    def __init__(self):
        super(fcn_autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(64 * 64 * 3, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 12),
            nn.ReLU(True),
            nn.Linear(12, 3))

        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 64 * 64 * 3),
            nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# maybe it can be smaller
class conv_autoencoder(nn.Module):
    def __init__(self):
        super(conv_autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 48, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 96, 4, stride=2, padding=1),  # medium: remove this layer
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # medium: remove this layer
            nn.ReLU(),
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.enc_out_1 = nn.Sequential(
            nn.Conv2d(24, 48, 4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.enc_out_2 = nn.Sequential(
            nn.Conv2d(24, 48, 4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def encode(self, x):
        h1 = self.encoder(x)
        return self.enc_out_1(h1), self.enc_out_2(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar


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


class Resnet(nn.Module):
    def __init__(self, fc_hidden1=1024, fc_hidden2=768, drop_p=0.3, CNN_embed_dim=256):
        super(Resnet, self).__init__()

        self.fc_hidden1, self.fc_hidden2, self.CNN_embed_dim = fc_hidden1, fc_hidden2, CNN_embed_dim

        # CNN architechtures
        self.ch1, self.ch2, self.ch3, self.ch4 = 16, 32, 64, 128
        self.k1, self.k2, self.k3, self.k4 = (5, 5), (3, 3), (3, 3), (3, 3)  # 2d kernal size
        self.s1, self.s2, self.s3, self.s4 = (2, 2), (2, 2), (2, 2), (2, 2)  # 2d strides
        self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (0, 0)  # 2d padding

        # encoding components
        resnet = models.resnet18(pretrained=False)
        modules = list(resnet.children())[:-1]  # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(resnet.fc.in_features, self.fc_hidden1)
        self.bn1 = nn.BatchNorm1d(self.fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.bn2 = nn.BatchNorm1d(self.fc_hidden2, momentum=0.01)

        self.fc3_mu = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)  # output = CNN embedding latent variables

        # Sampling vector
        self.fc4 = nn.Linear(self.CNN_embed_dim, self.fc_hidden2)
        self.fc_bn4 = nn.BatchNorm1d(self.fc_hidden2)
        self.fc5 = nn.Linear(self.fc_hidden2, 64 * 4 * 4)
        self.fc_bn5 = nn.BatchNorm1d(64 * 4 * 4)
        self.relu = nn.ReLU(inplace=True)

        # Decoder
        self.convTrans6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=self.k4, stride=self.s4,
                               padding=self.pd4),
            nn.BatchNorm2d(32, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.convTrans7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=self.k3, stride=self.s3,
                               padding=self.pd3),
            nn.BatchNorm2d(8, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        self.convTrans8 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=self.k2, stride=self.s2,
                               padding=self.pd2),
            nn.BatchNorm2d(3, momentum=0.01),
            nn.Sigmoid()  # y = (y1, y2, y3) \in [0 ,1]^3
        )

    def encode(self, x):
        x = self.resnet(x)  # ResNet
        x = x.view(x.size(0), -1)  # flatten output of conv

        # FC layers
        if x.shape[0] > 1:
            x = self.bn1(self.fc1(x))
        else:
            x = self.fc1(x)
        x = self.relu(x)
        if x.shape[0] > 1:
            x = self.bn2(self.fc2(x))
        else:
            x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3_mu(x)
        return x

    def decode(self, z):
        if z.shape[0] > 1:
            x = self.relu(self.fc_bn4(self.fc4(z)))
            x = self.relu(self.fc_bn5(self.fc5(x))).view(-1, 64, 4, 4)
        else:
            x = self.relu(self.fc4(z))
            x = self.relu(self.fc5(x)).view(-1, 64, 4, 4)
        x = self.convTrans6(x)
        x = self.convTrans7(x)
        x = self.convTrans8(x)
        x = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=True)
        return x

    def forward(self, x):
        z = self.encode(x)
        x_reconst = self.decode(z)

        return x_reconst


# # Dataset module
# 
# Module for obtaining and processing data. The transform function here normalizes image's pixels from [0, 255] to [-1.0, 1.0].
# 

# In[9]:


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
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __getitem__(self, index):
        x = self.tensors[index]

        if self.transform:
            # mapping images to [-1.0, 1.0]
            x = self.transform(x)

        return x

    def __len__(self):
        return len(self.tensors)


# # Training

# ## Initialize
# - hyperparameters
# - dataloader
# - model
# - optimizer & loss
# 

# In[10]:


# Training hyperparameters
num_epochs = 50
batch_size = 10000  # medium: smaller batchsize
learning_rate = 1e-3

# Build training dataloader
x = torch.from_numpy(train)
train_dataset = CustomTensorDataset(x)

train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

# Model
model_type = 'cnn'  # selecting a model type from {'cnn', 'fcn', 'vae', 'resnet'}
model_classes = {'resnet': Resnet(), 'fcn': fcn_autoencoder(), 'cnn': conv_autoencoder(), 'vae': VAE(), }
model = model_classes[model_type].cuda()

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate)

# ## Training loop

# In[ ]:


best_loss = np.inf
model.train()

qqdm_train = qqdm(range(num_epochs), desc=format_str('bold', 'Description'))
for epoch in qqdm_train:
    tot_loss = list()
    for data in train_dataloader:

        # ===================loading=====================
        if model_type in ['cnn', 'vae', 'resnet']:
            img = data.float().cuda()
        elif model_type in ['fcn']:
            img = data.float().cuda()
            img = img.view(img.shape[0], -1)

        # ===================forward=====================
        output = model(img)
        if model_type in ['vae']:
            loss = loss_vae(output[0], img, output[1], output[2], criterion)
        else:
            loss = criterion(output, img)

        tot_loss.append(loss.item())
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================save_best====================
    mean_loss = np.mean(tot_loss)
    if mean_loss < best_loss:
        best_loss = mean_loss
        torch.save(model, 'best_model_{}.pt'.format(model_type))
    # ===================log========================
    qqdm_train.set_infos({
        'epoch': f'{epoch + 1:.0f}/{num_epochs:.0f}',
        'loss': f'{mean_loss:.4f}',
    })
    # ===================save_last========================
    torch.save(model, 'last_model_{}.pt'.format(model_type))

# # Inference
# Model is loaded and generates its anomaly score predictions.

# ## Initialize
# - dataloader
# - model
# - prediction file

# In[ ]:


eval_batch_size = 200

# build testing dataloader
data = torch.tensor(test, dtype=torch.float32)
test_dataset = CustomTensorDataset(data)
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=eval_batch_size, num_workers=1)
eval_loss = nn.MSELoss(reduction='none')

# load trained model
checkpoint_path = 'last_model_{}.pt'.format(model_type)
model = torch.load(checkpoint_path)
model.eval()

# prediction file 
out_file = 'PREDICTION_FILE_01.csv'

# In[ ]:


anomality = list()
with torch.no_grad():
    for i, data in enumerate(test_dataloader):
        if model_type in ['cnn', 'vae', 'resnet']:
            img = data.float().cuda()
        elif model_type in ['fcn']:
            img = data.float().cuda()
            img = img.view(img.shape[0], -1)
        else:
            img = data[0].cuda()
        output = model(img)
        if model_type in ['cnn', 'resnet', 'fcn']:
            output = output
        elif model_type in ['res_vae']:
            output = output[0]
        elif model_type in ['vae']:  # , 'vqvae'
            output = output[0]
        if model_type in ['fcn']:
            loss = eval_loss(output, img).sum(-1)
        else:
            loss = eval_loss(output, img).sum([1, 2, 3])
        anomality.append(loss)
anomality = torch.cat(anomality, axis=0)
anomality = torch.sqrt(anomality).reshape(len(test), 1).cpu().numpy()

df = pd.DataFrame(anomality, columns=['Predicted'])
df.to_csv(out_file, index_label='Id')

# # Training statistics
# - Number of parameters
# - Training time on colab
# - Training curve of the bossbaseline model

# - Simple
#  - Number of parameters: 3176419
#  - Training time on colab: ~ 30 min
# - Medium
#  - Number of parameters: 47355
#  - Training time on colab: ~ 30 min
# - Strong
#  - Number of parameters: 47595
#  - Training time on colab:  4 ~ 5 hrs
# - Boss:  
#  - Number of parameters: 4364140
#  - Training time on colab: 1.5~3 hrs
# 
#  ![Screen Shot 2021-04-29 at 16.43.57.png](data:image/png;base64,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)
#