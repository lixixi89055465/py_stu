from mxnet.gluon import nn


def vgg_block(num_convs, channels):
    out = nn.Sequential()
    with out.name_scope():
        for __ in range(num_convs):
            out.add(nn.Conv2D(channels=channels, kernel_size=3,
                              padding=1, activation='relu'))
    out.add(nn.MaxPool2D(pool_size=2, strides=2))
    return out


# 我们实力化一个这样的块，里面有两个卷积层，每个卷积层输出通道是 128
from mxnet import nd

blk = vgg_block(2, 128)
blk.initialize()
x = nd.random.uniform(shape=(2, 3, 16, 16))
y = blk(x)
print(y.shape)


# 可以看到经过这样一个块后，长款会减半，通道也会改变。
# 然后我们定义如何将这些块堆积起来：
def vgg_stack(architecture):
    out=nn.Sequential()
    with out.name_scope():
        for(num_convs,channels) in architecture:
            out.add(vgg_block(num_convs,channels))
    return out
# 这里我们定义一个最简单的vgg结构，它有8个卷积层，和跟Alexnet一样的3个全连接层。
# 这个网络又称VGG11。

num_outputs=10
architecture=((1,64),(1,128),(2,256),(2,512),(2,512))
net=nn.Sequential()
with net.name_scope():
    net.add(vgg_stack(architecture))
    net.add(nn.Flatten())
    net.add(nn.Dense(4096,activation='relu'))
    net.add(nn.Dropout(.5))
    net.add(nn.Dense(4096,activation='relu'))
    net.add(nn.Dropout(.5))
    net.add(nn.Dense(num_outputs))

# print(net)
# 模型训练
# 这里跟Alexnet的训练代码一样：
import  sys
sys.path.append('..')
from  d2lzh_my import utils

def transform(data,label):
    # resize from 28*28 to 96*96
    data=utils.image.imresize(data,96,96)
    return utils.transform_mnist(data,label)

import d2lzh
batch_size=64
train_data,test_data=d2lzh.utils.load_data_fashion_mnist(
    batch_size,transform)

from mxnet import autograd
from mxnet import gluon
from mxnet import nd
from mxnet import init
ctx=utils.try_gpu()
print(ctx)
net.initialize(ctx=ctx,init=init.Xavier())
softmax_cross_entropy=gluon.loss.SoftmaxCrossEntropyLoss()
trainer=gluon.Trainer(
    net.collect_params(),'sgd',{'learning_rate':0.05} )

for epoch in range(1):
    train_loss=0.
    train_acc=0.
    for data,label in train_data:
        label=label.as_in_context(ctx)
        with autograd.record():
            output=net(data.as_in_context(ctx))
            loss=softmax_cross_entropy(output,label)
        loss.backward()
        trainer.step(batch_size)
        train_loss+=nd.mean(loss).asscalar()
        train_acc+=utils.accuracy(output,label)
    test_acc=utils.evaluate_accuracy(test_data,net,ctx)
    print('Epoch %d. Loss:%f; Train acc %f , Test acc %f' % (
        epoch, train_loss / len(train_data), train_acc.asscalar() / len(train_data), test_acc))


