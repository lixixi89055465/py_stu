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

