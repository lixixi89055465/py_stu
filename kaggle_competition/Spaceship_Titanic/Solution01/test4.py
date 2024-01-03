from sklearn import preprocessing
le=preprocessing.LabelEncoder()
print(le)
le.fit([1,2,3,4,8])
print('='*22)
print(le.classes_)
print('='*22)
print(le.transform([1, 1, 2, 8]))