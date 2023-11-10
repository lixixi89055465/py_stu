import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score

air_data = pd.read_csv("../../data/AirQualityUCI.csv").iloc[:, 2:-2]
print(air_data.head())
