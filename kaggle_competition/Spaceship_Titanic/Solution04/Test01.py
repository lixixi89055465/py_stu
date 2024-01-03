import numpy as np
import pandas as pd
a=pd.DataFrame(
    np.arange(0,10)
)
test_indices=np.random.rand(len(a))>0.2
print(a[test_indices])
print("1"*100)
print(a[~test_indices])

