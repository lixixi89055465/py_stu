from sklearn.cross_validation import KFold

def run_cv(X,y,clf_class,**kwargs):
    kf=KFold(len(y),n_folds=5,shuffle=True)
    y_pred=y.copy()
    # Iterate through folds
    for train_index,test_index in kf:
        X_train,X_test=X[train_index],X[test_index]
        y_train=y[train_index]
        # Initialize a classifier with key word arguments
        clf=clf_class(**kwargs)
        clf.fit(X_train,y_train)
        y_pred[test_index]=clf.predict(X_test)
    return y_pred