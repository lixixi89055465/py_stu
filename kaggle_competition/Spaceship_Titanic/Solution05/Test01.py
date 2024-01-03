import pandas as pd
import numpy as np

company = ["A", "B", "C"]
data = pd.DataFrame({
    "company": [company[x] for x in np.random.randint(0, len(company), 10)],
    "salary": np.random.randint(5, 50, 10),
    "age": np.random.randint(15, 50, 10)
})
print(data)
print(data.columns)
print('1' * 100)
group = data.groupby('company')
print(list(group))
print('2' * 100)
print(data.groupby('company').agg('mean'))
print('3' * 100)
print(data.groupby('company').agg('min'))
print(data.groupby('company').agg({'salary': 'median', 'age': 'mean'}))
