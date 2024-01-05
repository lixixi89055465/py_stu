import pandas as pd
import numpy as np

a = np.arange(1, 10).reshape(3, 3)
data = pd.DataFrame(a, index=["a", "b", "c"], columns=["one", "two", "three"])
print(data)

print(data.one.corr(data.two))
print(data.corr())
print(data.one.cov(data.two))
print(data.cov())
