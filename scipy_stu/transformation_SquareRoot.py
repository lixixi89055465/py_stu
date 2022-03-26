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
data=data_train
plot_data(data,'Age')
data['age_square']=data.Age**(1/2)
plot_data(data,'age_square')
