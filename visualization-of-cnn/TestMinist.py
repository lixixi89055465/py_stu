
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import torchvision
import torch.utils.data as Data
from sklearn.metrics import accuracy_score
from MyModel import ConvNet
# 准备训练用的MINIST数据集
train_data=torchvision.datasets.MNIST(
    root='./data/MNIST',#
    train=True,
    transform=torchvision.transforms.ToTensor(),#
    download=True
)
# 定义loader
train_loader=Data.DataLoader(
    dataset=train_data,
    batch_size=128,
    shuffle=True,
    num_workers=4
)
test_data=torchvision.datasets.MNIST(
    root='./data/MNIST',
    train=False, # 使用测试数据
    download=False
)
# 将测试数据压缩到 0-1
test_data_x=test_data.data.type(torch.FloatTensor)/255.0
test_data_x=torch.unsqueeze(test_data_x,dim=1).cuda()
test_data_y=test_data.targets

# 打印一下测试数据和训练数据的shape
print("test_data_x.shape:",test_data_x.shape)
print("test_data_y.shape:",test_data_y.shape)

for x,y in train_loader:
    print(x.shape)
    print(y.shape)
    break
model=ConvNet().cuda()
from tensorboardX import SummaryWriter
logger=SummaryWriter(log_dir='data/log')
# 获取优化器和损失函数
optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
loss_func=nn.CrossEntropyLoss()
log_step_interval=100
for epoch in range(5):
    print('epoch',epoch)
    for step,(x,y) in enumerate(train_loader):
        x,y=x.cuda(),y.cuda()
        predict=model(x)
        loss=loss_func(predict,y)
        loss.backward()
        optimizer.step()
        global_iter_num=epoch*len(train_loader)+step+1
        if global_iter_num%log_step_interval==0:
            print('global step{},loss:{:.2}'.format(global_iter_num,loss))
            logger.add_scalar('train_loss',loss.item(),global_step=global_iter_num)
            test_predict=model(test_data_x)
            _,predict_idx=torch.max(test_predict,1)
            predict_idx=predict_idx.cpu()
            acc=accuracy_score(test_data_y,predict_idx)
            logger.add_scalar('test accury',acc.item(),global_step=global_iter_num)
            img=torchvision.utils.make_grid(x.cpu(),nrow=12)
            logger.add_image("train image sample",img,global_step=global_iter_num)
            for name,param in model.named_parameters():
                logger.add_histogram(name,param.cpu().data.numpy(),global_iter_num)









