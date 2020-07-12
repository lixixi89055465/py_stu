import pandas
import numpy as np

print(pandas.__version__)
print(np.multiply([1, 2, 3], [5, 6, 7]))

x = np.array([ [0, 3, 4], [1, 6, 4]])
# x = np.array([ [3, 4], [4, 3] ])

print('默认参数 （矩阵整体元素平方和开根号，不保留矩阵二维特性):', np.linalg.norm(x))

print('矩阵整体元素平方和开根号，保留矩阵二维特性:', np.linalg.norm(x, keepdims=True))

print('矩阵每个行向量的2范数：',np.linalg.norm(x,axis=1,keepdims=True))
print('矩阵每个行向量的2范数：',np.linalg.norm(x,axis=1,keepdims=False))
print('矩阵每个列向量的2范数 :',np.linalg.norm(x,axis=0,keepdims=True))

print('矩阵1范数',np.linalg.norm(x,ord=1,keepdims=True))
print('矩阵2范数',np.linalg.norm(x,ord=2,keepdims=True))
# print('抉择3范数',np.linalg.norm(x,ord=3,keepdims=True))

