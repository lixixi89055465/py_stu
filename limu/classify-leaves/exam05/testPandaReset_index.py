import numpy as np
import pandas as pd

df = pd.DataFrame([('bird', 389.0),
				   ('bird', 24.0),
				   ('mammal', 80.5),
				   ('mammal', np.nan)],
				  index=['falcon', 'parrot', 'lion', 'monkey'],
				  columns=('class', 'max_speed'))

print(df)
print('0'*100)

print(df.reset_index())
print('1'*100)
print(df.reset_index(drop=True))

