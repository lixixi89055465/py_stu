import pandas as pd

s = "a,b,c,d\n1,2,3,4\n5,6,7,8\n9,10,11,12"
lst = []
for sep in s.split('\n'):
    lst.append(sep.split(','))
df = pd.DataFrame(lst[1:], columns=lst[0])
print(df)
