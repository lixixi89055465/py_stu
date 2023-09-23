import numpy as np
from machinelearn.linear_model_03.closed_form_sol.LinearRegression_CFSol import LinearRegressionClosedFormSol
from machinelearn.model_evaluation_selection_02.Polynomial_feature import PolynomialFeatureData
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def objective_fun(x):
    # return 0.5 * x ** 2 + x + 2
    return 0.5 * x ** 3 + 2 * x * x - 2.5 * x + 2


np.random.seed(42)  # 随机种子

n = 300  # 样本量
raw_x = np.sort(6 * np.random.rand(n, 1) - 3)  # 采样数据 【-3，3】，均匀分布
raw_y = objective_fun(raw_x) + 0.5 * np.random.randn(n, 1)  # 目标值，添加噪声
# raw_y = objective_fun(raw_x)

degree = [2, 3, 5, 10, 15, 20]  # 拟合多项式的最高阶次

# plt.figure(figsize=(15, 8))
from sklearn.model_selection import KFold
kfold=KFold(n_splits=10)#划分10折

for i, d in enumerate(degree):
    print('0' * 100)
    feature_obj = PolynomialFeatureData(raw_x, d, with_bias=False)  # 特征数据对象
    X_sample = feature_obj.fit_transform()  # 生成特征多项式
    train_mse, test_mse = [], []
    for j in range(1, 240):
        train_mse_,test_mse_=0.0,0.0
        for idx_train,idx_test in kfold.split(X_sample,raw_y):
            X_train,y_train=X_sample[idx_train],raw_y[idx_train] # 270个样本
            X_test, y_test= X_sample[idx_test], raw_y[idx_test]#30个样本

            lr_cfs = LinearRegressionClosedFormSol()
            lr_cfs.fit(X_train[:j, :], y_train[:j])  # 样本逐次增加
            y_test_pred = lr_cfs.predict(x_test=X_test)
            y_train_pred = lr_cfs.predict(X_train[:j, :])
            train_mse_+=np.mean((y_train_pred-y_train[:j])**2)
            test_mse_+=np.mean((y_test_pred - y_test) ** 2)
        train_mse.append(train_mse_/10)
        test_mse.append(test_mse_/10)

    # 可视化学习曲线
    plt.subplot(231 + i)
    plt.plot(train_mse, 'k-', lw=1, label='Train MSE')
    plt.plot(test_mse, 'r--', lw=1.5, label='Test MSE')
    plt.legend(frameon=False)
    plt.grid(ls=':')
    plt.xlabel("$Train Samples Size $", fontdict={'fontsize': 12})
    plt.ylabel("$MSE$", fontdict={'fontsize': 12})
    plt.title('Learning Curve with Degress {} '.format(d))
    # plt.axis([-3,3,0.5,20])
    # plt.axis([0,80,0,4])
    if i == 0:
        plt.axis([0, 80, 0, 10])
    if i == 1:
        plt.axis([0, 240, 0, 1])
    else:
        plt.axis([0, 240, 0, 2])
plt.show()
