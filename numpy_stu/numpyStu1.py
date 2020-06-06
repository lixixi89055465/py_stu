# coding: utf-8
# @Time    : 2020/6/6 7:55 AM
# @Author  : lixiang
# @File    : numpyStu1.py
import numpy as np

def fill_ndarray(t1):
    for i in range(t1.shape[1]):
        temp_col = t1[:, i]
        nan_num = np.count_nonzero(temp_col != temp_col)
        if nan_num != 0:
            temp_not_nan_col = temp_col[temp_col == temp_col]
            temp_col[np.isnan(temp_col)] = temp_not_nan_col.mean()
    return t1

if __name__=='__main__':
    t1=np.arange(24).reshape((4,6)).astype(float)
    t1[1,2:]=np.nan
    print(t1)
    t1=fill_ndarray(t1)
    print(t1)

