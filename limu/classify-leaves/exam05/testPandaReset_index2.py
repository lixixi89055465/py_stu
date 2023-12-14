import numpy as np
import pandas as pd

df = pd.DataFrame({'Country': ['China', 'China', 'India', 'India', 'America', 'Japan', 'China', 'India'],
				   'Income': [10000, 10000, 5000, 5002, 40000, 50000, 8000, 5000],
				   'Age': [50, 43, 34, 40, 25, 25, 45, 32]})
print(df)

print('0' * 100)
df_new = df.set_index('Country', drop=True, append=False, \
					  inplace=False, verify_integrity=False)
print(df_new)

df_new1 = df.set_index('Country', drop=False, append=True, inplace=False, verify_integrity=False)
print('1' * 100)
print(df_new1)
print('2' * 100)
print(df)
df_new1 = df.set_index('Country', drop=True, append=True, inplace=False, verify_integrity=False)
print('3' * 100)
print(df_new1)
print('4' * 100)

df_new = df.set_index('Country', drop=True, append=False, inplace=False, verify_integrity=False)
print('5' * 100)
print(df_new)
df_new01 = df_new.reset_index(drop=False)
print('6'*100)
print(df_new01)
df_new01 = df_new.reset_index(drop=True)
print('7'*100)
print(df_new01)