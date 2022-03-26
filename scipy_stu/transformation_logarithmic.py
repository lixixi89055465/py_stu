import matplotlib.pyplot as plt
import pylab
import scipy.stats as stat
import seaborn as sns
import pandas as pd
import numpy as np

def plot_data(data,feature):
    plt.figure(figsize=(10,8))
    plt.subplot(1,2,1)
    sns.histplot(data[feature],kde=True)
    plt.subplot(1,2,2)
    stat.probplot(data[feature],dist='norm',plot=pylab)
    plt.show()
data_train = pd.read_csv('./titanic_train.csv')
print(data_train.Age)
data=data_train
plot_data(data,'Age')
data['age_log']=np.log(data['Age'])
plot_data(data,'age_log')