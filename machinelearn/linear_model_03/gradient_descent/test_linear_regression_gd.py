'''


'''
import numpy as np
from machinelearn.linear_model_03.gradient_descent.LinearRegression_GD import LinearRegression_GradDesc
from sklearn.model_selection import train_test_split

np.random.seed(42)
X=np.random.rand(1000,6)#随机样本值
coeff=np.array([4.2,-2.5,7.8,3.7,-2.9,1.87])#模型系数
y=coeff.dot(X.T)+0.5*np.random.randn(1000)#目标函数值
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0,shuffle=True)

lr_grad=LinearRegression_GradDesc(max_epoch=10)

lr_grad.fit(X_train,y_train,X_test,y_test)
# lr_grad.plt_loss_curve()
