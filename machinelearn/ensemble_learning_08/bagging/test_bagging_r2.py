import numpy as np
import matplotlib.pyplot as plt
from machinelearn.decision_tree_04.decision_tree_R \
    import DecisionTreeRegression
from machinelearn.ensemble_learning_08.bagging.bagging_c_r \
    import BaggingClassifierRegression
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from machinelearn.decision_tree_04.decision_tree_R import DecisionTreeRegression
from machinelearn.decision_tree_04.decision_tree_C import DecisionTreeClassifier
from machinelearn.ensemble_learning_08.gradient.bagging_c_r import BaggingClassifierRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.metrics import accuracy_score

boston = load_boston()
X, y = boston.data, boston.target
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

base_estimator = DecisionTreeRegression(max_bins=50, max_depth=8)

# model = BaggingClassifierRegression( \
#     base_estimator=base_estimator, n_estimators=20, task='r')
model = BaggingClassifierRegression( \
    base_estimator=base_estimator, n_estimators=20, task='r', OOB=True)
model.fit(X, y)
idx = np.argsort(y_test)
y_hat = model.predict(X_test)

plt.figure(figsize=(7, 5))
plt.plot(y_hat[idx], 'r-', lw=1.5, label="Bagging Prediction")
plt.plot(y_test[idx], 'k-', label='Test True Values')
plt.legend(frameon=False)
plt.xlabel("X", fontsize=12)
plt.ylabel("Y", fontsize=12)
plt.grid(':')
plt.title("Bagging (20 estimators) VS Regressor (R2=%.5f)" % r2_score(y_test, y_hat), fontsize=14)
plt.show()
