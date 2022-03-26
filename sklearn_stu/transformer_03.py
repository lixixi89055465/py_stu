import numpy as np
import math
import scipy

X=np.random.rand(1000,1)
mu=0
sigma=1
Y=mu+math.sqrt(2)*sigma*scipy.special.erfinv(2*X-1)
