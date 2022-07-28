import hiddenlayer as hl
import torch
import torchvision
import torch.nn as nn
import torch.utils.data as Data
from sklearn.metrics import accuracy_score


from MyModel import ConvNet
import time
# 记录训练过程的指标
history=hl.History()
# 使用canvas 进行可视化
canvas=hl.Canvas()
# 获取优化器和损失函数
MyConvNet=ConvNet()
optimizer=torch.optim.Adam(MyConvNet.parameters(),lr=3e-4)
loss_func=nn.CrossEntropyLoss()
log_step_interval=100 # 记录的步数间隔
train_data=torchvision.datasets.MNIST(
    root='./data/MNIST',#
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True
)
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
test_data_x=torch.unsqueeze(test_data_x,dim=1)
test_data_y=test_data.targets

# 打印一下测试数据和训练数据的shape
print("test_data_x.shape:",test_data_x.shape)
print("test_data_y.shape:",test_data_y.shape)

for epoch in range(5):
    print('epoch',epoch)
    for step,(x,y) in enumerate(train_loader):
        predict=MyConvNet(x)
        loss=loss_func(predict,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_iter_num=epoch*len(train_loader)+step+1
        if global_iter_num%log_step_interval==0:
            print('global_step:{},loss:{:.2}'.format(global_iter_num,loss.item()))
            test_predict=MyConvNet(test_data_x)
            _,predict_idx=torch.max(test_predict,1)
            acc=accuracy_score(test_data_y,predict_idx)
            history.log((epoch,step),
                        train_loss=loss,
                        test_acc=acc,
                        hidden_weight=MyConvNet.fc[2].weight)
            # 可视化
            with canvas:
                canvas.draw_plot(history['train_loss'])
                canvas.draw_plot(history['test_acc'])
                canvas.draw_image(history['hidden_weight'])





