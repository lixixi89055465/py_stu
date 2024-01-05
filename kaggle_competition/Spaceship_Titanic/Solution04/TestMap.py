import pandas as pd
import numpy as np

s = pd.Series(['cat', 'dog', np.nan, 'rabbit'])
print(s)

print('0' * 100)
print(s.map({'cat': 'kitten', 'dog': 'puppy'}))

print('1' * 100)
s.map('I am a {}'.format)

print('2' * 100)
print(s.map('I am a {}'.format, na_action='ignore'))
print('3' * 100)
data = pd.DataFrame(
    np.array([['任盈盈', 0], ['任我行', 1], ['东方兄', np.nan], ['令狐冲', 1]]),
    columns=['name', 'gender'])
print(data)
# print('4' * 100)
# data['gender'] = data['gender'].map({'1': '男', '0': '女'})
# print(data)
print('5' * 100)
data.replace({'gender': {'1': '男', '0': '女'}})
print(data)
print('6' * 100)


def handle_gender1(X):
    if X == '1':
        return '男'
    elif X == '0':
        return '女'
    else:
        return '未知'


print(data['gender'].map(handle_gender1))
def handle_gender2(X):
    if X == '1':
        return '男'
    elif X == '0':
        return '女'
    return '未知'


print('7'*100)
print(data['gender'].map(handle_gender2))
print('8'*100)
print(data['gender'].map(handle_gender2,na_action='ignore'))


