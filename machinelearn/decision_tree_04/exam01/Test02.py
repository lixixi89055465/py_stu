import numpy as np

num_class = 4
arr = [2, 3, 1]
a=np.eye(num_class)
print(a)
print(a[arr])
import pandas as pd

df = pd.DataFrame({'carduser_id': [12345,223432,343424],
                   'gender': ['男','男','女']})
dummies = pd.get_dummies(df)
print('2'*100)
print(dummies)
dummies = dummies.rename(columns={'gender_女':'female','gender_男':'male'})
print('3'*100)
print(dummies)