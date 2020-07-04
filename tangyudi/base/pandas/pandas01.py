import pandas
food_info=pandas.read_csv("food_info.csv")
print(type(food_info))
print(food_info.dtypes)
print(food_info.head())
first_rows=food_info.head(3)
print(first_rows)
print(food_info.tail(3))

print(food_info.columns)

print(food_info.shape)
print('*'*100)
print(food_info.loc[0])
food_info['NDB_No'].print
