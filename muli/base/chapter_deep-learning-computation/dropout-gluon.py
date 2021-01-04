# 丢弃法 -- 使用 gluon
# 本章介绍 如何使用 gluon
# 定义模型并添加批量归一化
# 有了gluon,我们模型的定义工作就简单了很多 。我们只需要在全连接层后添加gluon.nn.Dropout 并指定元素丢弃概率
# 一般情况下，我们推荐把靠近输入层的元素丢弃概率设的小一点。这个试验中，我们把第一层全连接后的元素
# 设置为0.2，把第二层全连接后的元素的丢弃概率设置为 0.5


from mxnet.gluon import nn


def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad


def accuracy(output, label):
    return nd.mean(output.argmax(axis=1) == label)


net = nn.Sequential()
drop_prob1 = 0.2
drop_prob2 = 0.5

with net.name_scope():
    net.add(nn.Flatten())
    # 第一层全连接。
    net.add(nn.Dense(256, activation='relu'))
    # 在第一层全连接后添加丢弃层
    net.add(nn.Dropout(drop_prob1))
    # 第二层全连接
    net.add(nn.Dense(256, activation='relu'))
    net.add(nn.Dropout(drop_prob2))
    net.add(nn.Dense(10))
net.initialize()

# 读取数据
import sys

sys.path.append('..')
from d2lzh import utils
from mxnet import nd
from mxnet import autograd
from mxnet import gluon

batch_size = 256
train_data, test_data = utils.load_data_fashion_mnist(batch_size)

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})

for epoch in range(5):
    train_loss = 0.
    train_acc = 0.
    for data, label in train_data:
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(batch_size)
        train_loss += nd.mean(output).asscalar()
        train_acc += accuracy(output, label.astype("float32"))
    test_acc = utils.evaluate_accuracy(test_data,net)
    print('Epoch %d. Loss:%f; Train acc %f , Test acc %f' % (
        epoch, train_loss / len(train_data), train_acc.asscalar() / len(train_data), test_acc))
