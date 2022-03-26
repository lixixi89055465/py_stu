from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
quantile_transformer = preprocessing.QuantileTransformer(random_state=0)
X_train_trans = quantile_transformer.fit_transform(X_train)
X_test_trans = quantile_transformer.transform(X_test)
a=np.percentile(X_train[:, 0], [0, 25, 50, 75, 100])
print(a)
print('1'*100)


b=np.percentile(X_test[:,0],[0,25,50,75,100])
print(b)
print('2'*100)

# pt=preprocessing.PowerTransformer(method='box-cox',standardize=False)
pt=preprocessing.PowerTransformer(method='yeo-johnson',standardize=False)
X_lognormal=np.random.RandomState(616).lognormal(size=(3,3))
print(X_lognormal)
a=pt.fit_transform(X_lognormal)
print(a)
print('3'*100)
quantile_transformer=preprocessing.QuantileTransformer(
    output_distribution='normal',random_state=0
)
X_trans=quantile_transformer.fit_transform(X)
print(quantile_transformer.quantiles_)


print('4'*100)
from sklearn.preprocessing import QuantileTransformer
rng = np.random.RandomState(0)
X = np.sort(rng.normal(loc=0.5, scale=0.25, size=(25, 1)), axis=0)
qt = QuantileTransformer(n_quantiles=10, random_state=0)
a=qt.fit_transform(X)
print(a)
print('5'*100)
import numpy as np
from sklearn.preprocessing import PowerTransformer
pt=PowerTransformer()
data=[[1,2],[3,2],[4,5]]
a=pt.fit(data )
print(a)
print(pt.lambdas_)
print(pt.transform(data ))