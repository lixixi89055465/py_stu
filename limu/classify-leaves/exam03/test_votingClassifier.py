import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

X, y = datasets.make_moons(n_samples=500, noise=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier

# voting_clf = VotingClassifier(
# 	estimators=[
# 		('log_clf', LogisticRegression()),
# 		('svm_clf', SVC()),
# 		('dt_clf', DecisionTreeClassifier(random_state=666)),
# 	],
# 	voting='hard'
# )
# voting_clf.fit(X_train, y_train)
# print(voting_clf.score(X_test, y_test))

voting_clf = VotingClassifier(
	estimators=[
		('log_clf', LogisticRegression()),
		('svm_clf', SVC(probability=True)),
		('dt_clf', DecisionTreeClassifier(random_state=666)),
	],
	voting='soft'
)
voting_clf.fit(X_train, y_train)
print(voting_clf.score(X_test, y_test))

