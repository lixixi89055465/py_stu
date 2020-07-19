# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# %%
data = pd.read_csv('creditcard.csv')

#%%
data['Amount'].values.reshape(-1,1)
#%%

from sklearn.preprocessing import StandardScaler

data['normAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
data = data.drop(['Time', 'Amount'], axis=1)
X = data.loc[:, data.columns != 'Class']
y = data.loc[:, data.columns == 'Class']
number_records_fraud = len(data[data.Class == 1])
fraud_indices = np.array(data[data.Class == 1].index)
normal_indices = data[data.Class == 0].index
random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace=False)
# %%
# type(random_normal_indices)
random_normal_indices = np.array(random_normal_indices)
type(random_normal_indices)
# %%
under_sample_indices = np.concatenate([fraud_indices, random_normal_indices])
# %%
# print(under_sample_indices)

# %%
under_sample_data = data.iloc[under_sample_indices, :]
X_undersample = under_sample_data.loc[:, under_sample_data.columns != 'Class']
y_undersample = under_sample_data.loc[:, under_sample_data.columns == 'Class']
print('Percentage of normal transaction:',
      len(under_sample_data[under_sample_data.Class == 0]) / len(under_sample_data))
print('Percentage of fraud transactions:',
      len(under_sample_data[under_sample_data.Class == 1]) / len(under_sample_data))
print('Total number of transactions in resampled data:', len(under_sample_data))
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import confusion_matrix, recall_score, classification_report
# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = train_test_split(
    X_undersample,
    y_undersample,
    test_size=0.3,
    random_state=0
)


def printing_Kfold_scores(x_train_data, y_train_data):
    # fold=KFold(len(y_train_data),5,shuffle=False)
    fold = KFold(5, shuffle=False)
    # Different C parameters
    c_param_range = [0.01, 0.1, 1, 10, 100]
    results_table = pd.DataFrame(index=range(len(c_param_range), 2), columns=['C_parameter', 'Mean recall score'])
    results_table['C_parameter'] = c_param_range
    # the k-fold will give 2 lists train_indices=indices[0],test_indices=indices[1]
    j = 0
    for c_param in c_param_range:
        print('-' * 50)
        print('C parameter:', c_param)
        print('-' * 50)
        print('')
        recall_accs = []
        for iteration, indices in enumerate(fold.split(x_train_data)):
            # call the logistic regression model with a certain C parameter
            lr = LogisticRegression(C=c_param, penalty='l1',solver='liblinear')
            lr.fit(x_train_data.iloc[indices[0], :], y_train_data.iloc[indices[0], :].values.ravel())
            y_pred_undersample = lr.predict(x_train_data.iloc[indices[1], :].values)
            recall_acc = recall_score(y_train_data.iloc[indices[1], :].values, y_pred_undersample)
            recall_accs.append(recall_acc)
            print('Iteration  ', iteration, ': recall score =', recall_acc)
        results_table.loc[j:'Mean recall score '] = np.mean(recall_accs)
        j += 1
        print('')
        print('Mean recall score', np.mean(recall_accs))
        print('')
    best_c = results_table.loc[results_table['Mean recall score'].idxmax()]['C_parameter']
    print('*' * 100)
    print('Best model to choose from cross validation is with C parameter =', best_c)
    print('*' * 100)
    return best_c


# %%
best_c = printing_Kfold_scores(X_train_undersample, y_train_undersample)

# %%
