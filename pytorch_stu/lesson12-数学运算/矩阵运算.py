import torch

a = torch.rand(3, 4)
b = torch.rand(4)
print(a + b)
print(torch.add(a, b))

print(torch.all(torch.eq(a - b, torch.sub(a, b))))

print(torch.all(torch.eq(a * b, torch.mul(a, b))))

print(torch.all(torch.eq(a / b, torch.div(a, b))))
# matmul
a = torch.ones([2, 2]) * 3
print(a)
b = torch.ones([2, 2], dtype=torch.float32)
print(b)
# c = torch.mm(a, b)
# print(c)
print('-' * 10)
print(torch.matmul(a, b))

print(a @ b)
# example
a = torch.rand(4, 784)
x = torch.rand(4, 784)
w = torch.rand(512, 784)
# print((x @ w.T).shape)

a = torch.rand(4, 3, 28, 64)
b = torch.rand(4, 3, 64, 32)
# print(torch.mm(a, b).shape)
print(torch.matmul(a, b).shape)
b = torch.rand(4, 1, 64, 32)
print(torch.matmul(a, b).shape)
# b=torch.rand(4,64,32)
# print(torch.matmul(a, b).shape)
a = torch.full([2, 2], 3)
print(a.pow(2))

print(a ** 2)
aa=a**2
print(aa.sqrt())
print(aa.rsqrt())
print(aa**0.5)
# exp log
a=torch.exp(torch.ones(2,2))
print(a)
print(torch.log(a))
a=torch.tensor(3.24)
print(a.floor(), a.ceil(), a.trunc(), a.frac())
a=torch.tensor(3.499)
print(a.round())
a=torch.tensor(3.5)
print(a.round())
grad=torch.rand(2,3)*15
print(grad)
print(grad.max())
print(grad.median())
# 最小值是10
print(grad.clamp(10))
# 限定值的范围
print(grad.clamp(5, 10))

print('max:')
grad=torch.rand(3,4)
print(grad)
print(grad.max(dim=1))
prob=torch.argmax(grad,dim=1)
print(grad.max(dim=1)[0]>0.7)
index=grad.max(dim=1)[0]>0.7
print(prob[index])
