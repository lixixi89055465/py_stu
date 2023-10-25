import numpy as np
import matplotlib.pyplot as plt
from machinelearn.decision_tree_04.decision_tree_R \
    import DecisionTreeRegression
from machinelearn.ensemble_learning_08.bagging.bagging_c_r \
    import BaggingClassifierRegression
from sklearn.metrics import r2_score

f = lambda x: 0.5 * np.exp(-(x + 3) ** 2) + \
              np.exp(-x ** 2) + 1.5 * np.exp(-(x - 3) ** 2)
np.random.seed(0)
N = 200
X = np.random.rand(N) * 10 - 5
X = np.sort(X)
y = f(X) + 0.05 * np.random.randn(N)
X = X.reshape(-1, 1)

base_estimator = DecisionTreeRegression(max_bins=30, max_depth=8)

model = BaggingClassifierRegression( \
    base_estimator=base_estimator, n_estimators=100, task='r')
model.fit(X, y)

X_test = np.linspace(X.min() * 1.1, X.max() * 1.1, 1000).reshape(-1, 1)
y_bagging_hat = model.predict(X_test)

base_estimator.fit(X, y)
y_cart_hat = base_estimator.predict(X_test)

plt.figure(figsize=(7, 5))
plt.scatter(X, y, s=10, c='k', label='Raw Data')
plt.plot(X_test, f(X_test), 'k-', lw=1.5, label="True F(X)")
plt.plot(X_test, y_bagging_hat, 'r-', label='Bagging (R2=%.5f) ' % \
                                            r2_score(f(X_test), y_bagging_hat))
plt.plot(X_test, y_cart_hat, 'r-', label='Cart (R2=%.5f) ' % \
                                         r2_score(f(X_test), y_cart_hat))

plt.legend(frameon=False)
plt.xlabel("X", fontsize=12)
plt.ylabel("Y", fontsize=12)
plt.grid(':')
plt.title("Bagging (100 estimators) VS Cart Regressor ", fontsize=14)
plt.show()
