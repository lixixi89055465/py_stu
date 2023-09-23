import numpy as np
from machinelearn.linear_model_03.closed_form_sol.LinearRegression_CFSol import LinearRegressionClosedFormSol
from machinelearn.model_evaluation_selection_02.Polynomial_feature import PolynomialFeatureData
import matplotlib.pyplot as plt


def objective_fun(x):
    # return 0.5 * x ** 2 + x + 2
    return 0.5 * x ** 3 + 2 * x * x - 2.5 * x + 2


np.random.seed(42)  # 随机种子

n = 100  # 样本量
raw_x = np.sort(6 * np.random.rand(n, 1) - 3)  # 采样数据 【-3，3】，均匀分布
raw_y = objective_fun(raw_x) + 0.5 * np.random.randn(n, 1)  # 目标值，添加噪声
# raw_y = objective_fun(raw_x)

plt.figure(figsize=(15, 8))
degree = [1, 2, 5, 10, 15, 20]  # 拟合多项式的最高阶次
for i, d in enumerate(degree):
    print('0' * 100)
    feature_obj = PolynomialFeatureData(raw_x, d, with_bias=False)  # 特征数据对象
    X_sample = feature_obj.fit_transform()  # 生成特征多项式
    lr_cfs = LinearRegressionClosedFormSol()  # 采用线性回归求解多项式
    lr_cfs.fit(X_sample, raw_y)  # 求解多项式回归系数
    theta = lr_cfs.get_params()  # 获取系数
    print('degree: %d, theta is ' % d, theta[0].reshape(-1)[::-1], theta[1])
    y_train_pred = lr_cfs.predict(X_sample)  # 在训练集上的预测

    # 测试样本采样
    x_test_raw = np.linspace(-3, 3, 150)  # 测试数据
    y_test = objective_fun(x_test_raw)  # 测试数据真值
    feature_obj = PolynomialFeatureData(x_test_raw, degree=d, with_bias=False)  # 特征数据对象
    X_test = feature_obj.fit_transform()  # 生成多项式特征测试数据
    y_test_pred = lr_cfs.predict(X_test)  # 模型在测试样本上的预测值

    # 可视化多项式拟合曲线
    plt.subplot(231 + i)
    plt.scatter(raw_x, raw_y, edgecolors='k', s=10, label='Raw data')
    plt.plot(x_test_raw, y_test, 'k-', lw=1, label='Objective Fun')
    plt.plot(x_test_raw, y_test_pred, 'r--', lw=1.5, label='Polynomial Fit')

    plt.legend(frameon=False)
    plt.grid(ls=':')
    plt.xlabel("$x$", fontdict={'fontsize': 12})
    plt.ylabel("$y(x)$", fontdict={'fontsize': 12})

    test_ess = (y_test_pred.reshape(-1) - y_test.reshape(-1)) ** 2  # 误差平方
    test_mse, test_std = np.mean(test_ess), np.std(test_ess)

    train_ess = (y_train_pred.reshape(-1) - raw_y.reshape(-1)) ** 2  # 误差平方
    train_mse, train_std = np.mean(train_ess), np.std(train_ess)

    print(y_test_pred.shape, y_test.shape)
    plt.title('Degree {} Test MSE = {:.2e}(+/-{:.2e}) \n Train Mse = {:.2e}(+/-{:.2e}'
              .format(d, test_mse, test_std, train_mse, train_std), fontdict={'fontsize': 8})
plt.show()
