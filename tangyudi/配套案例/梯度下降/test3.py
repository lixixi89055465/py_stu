import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import os
path='LogiReg_data.txt'
pdData=pd.read_csv(path,header=None,names=['Exam 1','Exam 2','Admitted'])
print(pdData.head())

positive=pdData[pdData['Admitted']==1]
negative=pdData[pdData['Admitted']==0]
print(positive.head())


fig,ax=plt.subplots(figsize=(10,5))
ax.scatter(positive['Exam 1'],positive['Exam 2'],s=50,c='b',marker='o',label='Admitted')
ax.scatter(negative['Exam 1'],negative['Exam 2'],s=30,c='r',marker='x',label='Not Admitted')

# ax.legend()
# ax.set_xlabel('Exam 1 Score')
# ax.set_ylabel('Exam 2 Score')
