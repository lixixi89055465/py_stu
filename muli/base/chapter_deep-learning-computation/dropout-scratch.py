# 丢弃法的实现
# 丢弃发很容易实现，例如下面这样。这里的标量drop_probability 定义了一个X（NDArray类 )
# 中任何一个元素被丢弃的概率
from mxnet.gluon import nn

print(nn.Dense)
from mxnet import nd

def SGD(params,lr):
    for param in params:
        param[:]=param -lr *param.grad


def accuracy(output,label):
    return nd.mean(output.argmax(axis=1)==label)





def dropout(X, drop_probalility):
    keep_probability = 1 - drop_probalility
    assert 0 <= keep_probability <= 1
    # 这种情况下把全部元素丢弃掉。
    if keep_probability == 0:
        return X.zeros_like()
    # 随机选择一部分该层的输出作为丢弃元素。
    mask = nd.random.uniform(
        0, 1.0, X.shape, ctx=X.context) < keep_probability
    # 保证 E[dropout(x)]==X
    scale = 1 / keep_probability
    return mask * X * scale


# 运行几个实例来验证一下
A = nd.arange(20).reshape((5, 4))
print(dropout(A, 0.0))
print(dropout(A, 0.5))
print(dropout(A, 1))
# 丢弃发的本质
# 了解了丢弃发的概念与实现，那你可能对它的本质产生了好奇。
# 如果你了解集成学习，你可能知道它在提升弱分类器准确率上的威力。一般来说，在集成学习里，
# 我们可以对训练数据集有放回地采样若干次并分别训练若干个不同的分类起；测试时，把这些分类器
# 的结果集成一个最终的分类器结果。
# 事实上，丢弃法在模拟集成学习。
# 数据获取
# 我们继续使用FashionMNIST数据集
import sys

sys.path.append('..')
import d2lzh
from d2lzh import utils

batch_size = 256
train_data, test_data = utils.load_data_fashion_mnist(batch_size)
print(train_data)

# 含有两个隐藏层的多层感知机
# 多层感知机。这里我们定义一个包含两个隐藏层的模型，两个隐含层都输出256个
# 我们定义激活函数Relu并直接使用Gluon提供的交叉熵损失函数

num_inputs = 28 * 28
num_outputs = 10

num_hidden1 = 256
num_hidden2 = 256
weight_scale = 0.01
W1 = nd.random_normal(shape=(num_inputs, num_hidden1), scale=weight_scale)
b1 = nd.zeros(num_hidden1)

W2 = nd.random_normal(shape=(num_hidden1, num_hidden2), scale=weight_scale)
b2 = nd.zeros(num_hidden2)

W3 = nd.random_normal(shape=(num_hidden2, num_outputs), scale=weight_scale)
b3 = nd.zeros(num_outputs)
params = [W1, b1, W2, b2, W3, b3]
for param in params:
    param.attach_grad()

# 定义包含丢弃层的模型
# 我们的模型就是将层（全连接）和激活函数（Relu)串起来，并在应用激活函数后添加丢弃层。
# 每个丢弃层的元素丢弃概率可以分别设置。一般情况下，我们推荐把更靠近输入层的元素丢弃概率设置小一点。
# 这个试验中,我们把第一层全连接后的元素丢弃概率设置为0.2,并把第二层全连接后的元素丢弃概率设为0.5
drop_prob1 = 0.2
drop_prob2 = 0.5


def net(X):
    X = X.reshape((-1, num_inputs))
    # 第一层全连接。
    h1 = nd.relu(nd.dot(X, W1) + b1)
    # 第一层全连接后添加丢弃层
    h1 = dropout(h1, drop_prob1)
    # 第二层全连接 。
    h2 = nd.relu(nd.dot(h1, W2) + b2)
    # 在第二层全连接后添加丢弃层
    h2 = dropout(h2, drop_prob2)
    return nd.dot(h2, W3) + b3


# 训练
# 训练和之前一样
from mxnet import autograd
from mxnet import gluon

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

learning_rate = .5
for epoch in range(5):
    train_loss = 0.
    train_acc = 0.
    for data, label in train_data:
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        SGD(params, learning_rate / batch_size)
        train_loss += nd.mean(loss).asscalar()
        train_acc += accuracy(output, label.astype("float32"))
    test_acc = utils.evaluate_accuracy(test_data, net)
    print('Epoch %d. Loss:%f; Train acc %f , Test acc %f' % (
        epoch, train_loss / len(train_data), train_acc.asscalar() / len(train_data), test_acc))
