'''


'''
import numpy as np
from machinelearn.linear_model_03.gradient_descent.LinearRegression_GD import LinearRegression_GradDesc
from sklearn.model_selection import train_test_split

np.random.seed(42)
X=np.random.rand(1000,6)#随机样本值
coeff=np.array([4.2,-2.5,7.8,3.7,-2.9,1.87])#模型系数
y=coeff.dot(X.T)+0.1*np.random.randn(1000)#目标函数值
# y=coeff.dot(X.T)
# y=X.dot(coeff)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=5,shuffle=True)

# lr_grad=LinearRegression_GradDesc()
lr_gd=LinearRegression_GradDesc(alpha=0.1, batch_size=1, max_epoch=300)
lr_gd.fit(X_train, y_train, X_test, y_test)
theta=lr_gd.get_params()
print(theta)
y_test_pred=lr_gd.predict(X_test)
lr_gd.plt_predict(y_test,y_test_pred)

lr_gd.plt_loss_curve()
