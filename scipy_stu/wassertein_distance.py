'''

'''
from scipy.stats import wasserstein_distance

a=wasserstein_distance([0,1,3],[5,6,8])
print(a)
a=wasserstein_distance([0,1,3],[1,0,3])
print(a)
b=wasserstein_distance([0,1],[0,1],[3,1],[2,2])
print(b)

c=wasserstein_distance([3.4, 3.9, 7.5, 7.8], [4.5, 1.4],
                     [1.4, 0.9, 3.1, 7.2], [3.2, 3.5])
print(c)