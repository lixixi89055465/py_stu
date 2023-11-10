import numpy as np
import matplotlib.pyplot as plt
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
# from machinelearn.decision_tree_04.decision_tree_C import DecisionTreeClassifier
# from machinelearn.ensemble_learning_08.gradient.bagging_c_r import BaggingClassifierRegression
from machinelearn.ensemble_learning_08.randomforest.rf_classifier_regressor import RandomForestClassifierRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
from sklearn.metrics import accuracy_score
import seaborn as sns

# boston = load_boston()
air = pd.read_csv('../../data/airquality.csv', encoding='GBK').dropna()
X, y = np.asarray(air.iloc[:, 3:]), np.asarray(air.iloc[:, 3])
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

# base_estimator = DecisionTreeRegression(max_bins=50, max_depth=8)

# model = BaggingClassifierRegression( \
#     base_estimator=base_estimator, n_estimators=20, task='r')
base_estimator = DecisionTreeRegressor(max_depth=10)
model = RandomForestClassifierRegressor( \
    base_estimator=base_estimator, n_estimators=50, task='r', OOB=True, \
    feature_sampling_rate=0.5,feature_importance=True)
model.fit(X_train, y_train)
idx = np.argsort(y_test)
y_pred = model.predict(X_test)

plt.figure(figsize=(7, 5))
plt.plot(y_pred[idx], 'r-', lw=1.5, label="Bagging Prediction")
plt.plot(y_test[idx], 'k-', label='Test True Values')
plt.legend(frameon=False)
plt.xlabel("X", fontsize=12)
plt.ylabel("Y", fontsize=12)
plt.grid(':')
plt.title("Air Quality (R2 = %.5f, MSE =%.5f)" % \
          (r2_score(y_test, y_pred), ((y_test - y_pred) ** 2).mean()), \
          fontsize=14)
plt.show()

plt.figure(figsize=(9, 5))
data_pd = pd.DataFrame([air.columns[3:], model.feature_importance_scores]).T
data_pd.columns = ['Feature Names', 'Importance']
sns.barplot(x='Importance', y='Feature Names', data=data_pd)
plt.title('AirQuality Dataset Feature Importance Scores', fontdict={'fontsize': 14})
plt.grid(ls=":")
plt.show()
