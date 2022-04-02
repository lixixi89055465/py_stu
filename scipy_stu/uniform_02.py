
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
x=stats.uniform(0,1).rvs(10000)
sns.distplot(x,kde=False,norm_hist=True)
print('over')