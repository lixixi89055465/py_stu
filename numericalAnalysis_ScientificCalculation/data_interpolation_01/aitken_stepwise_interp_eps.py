# -*- coding: utf-8 -*-
# @Time    : 2023/9/24 下午1:26
# @Author  : nanji
# @Site    : 
# @File    : aitken_stepwise_interp_eps.py
# @Software: PyCharm 
# @Comment :
import numpy as np
import matplotlib.pyplot as plt



class AitkenStepWiseInterpolationWithEpsilon:
    '''
      艾特肯逐步插值基本思想时K+1次插值多项式可由两个k次插值多项式得出
      带精度要求，未必逐步递推到最后一个多项式,只要达到精度要求即可，不在进行递推
      '''

    def __init__(self, x, y,eps=1e-3):
        '''
        艾特肯逐步插值基本思想时K+1次插值多项式可由两个k次插值多项式得出
        @param x:
        @param y:
        '''
        self.x = np.asarray(x, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)  # 类型转换,数据结构采用array
        if len(self.x) > 1 and len(self.x) == len(self.y):
            self.n = len(self.x)  # 已知离散数据点的个数
        else:
            raise ValueError("插值数据(x,y)维度不匹配!")
        self.eps=eps # 插值点的精度要求
        self.y0 = None  # 所求插值点的值，单个值或向量
        self.recurrence_num=None # 存储每个插值点的递推次数


    def fit_interp(self,x0):
        '''
        核心算法：根据给定的插值点x0,并根据精度要求，逐步递推
        @return:
        '''
        x0=np.asarray(x0,dtype=np.float)
        y_0=np.zeros(len(x0)) # 用于存储对应x0的插值
        self.recurrence_num=[]
        for k in range(len(x0)):#用于存储对应x0的插值
            val_next = np.zeros(self.n)# 用于存储下一列递归多项式的值
            val_before = np.copy(self.y)  # 用于存储上一列递推多项式的值
            tol,i=1,0# 初始精度
            for i in range(self.n - 1):
                # 针对每一个数据点
                for j in range(i + 1, self.n):
                    val_next[j] = (val_before[j] * (x0[k] - self.x[i]) - val_before[i] * (x0[k] - self.x[j])) \
                                   / (self.x[j] - self.x[i])
                tol=np.abs(val_before[i+1]-val_next[i+1])# 精度更新
                val_before[i+1:]= val_next[i+1:]  # 多项式的递推，下一列赋值给上一列
                if tol<=self.eps:# 满足精度要求，跳出i循环，不再进行递推
                    break
            y_0[k]=val_next[i+1]#满足精度要求的插值存储
            self.recurrence_num.append(i+1)

        self.y0=y_0# 计算完毕，负值给类属性变量，供用户调用
        return y_0

    def plt_interpolation(self, x0=None, y0=None):
        '''
        可视化插值图像和所求的插值点
        :return:
        '''
        plt.figure(figsize=(8,6))
        plt.plot(self.x,self.y,'ro',label='Interpolation base point ')
        xi=np.linspace(min(self.x),max(self.x),100)# 模拟100各值
        yi=self.fit_interp(xi)
        plt.plot(xi,yi,'b--',label='Interpolation polynomial ')
        if x0 is not None and y0 is not None:
            plt.plot(x0,y0,'g*',label='Interpolation point values ')
        plt.legend()
        plt.xlabel("x",fontdict={'fontsize':12})
        plt.ylabel("y",fontdict={'fontsize':12})
        # avg_rec=np.round(np.mean(self.recurrence_num),2)
        # print(self.recurrence_num)
        # plt.title('Aiten interpolation avg_recurence times is %.1f(100 points ) with eps= %.1e' %(avg_rec,self.eps),
        #           fontdict={'fontsize':14})
        plt.grid(ls=':')
        plt.show()






