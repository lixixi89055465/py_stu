# test.py
import  numpy as np
from scipy.spatial.distance import  pdist, squareform

if __name__ == "__main__":
    a = np.array([[1,1,1,1],[1,0,0,0]])
    b = np.array([[1,0,0,1],[1,1,0,0]])

    c = (a == 1)
    print(c)

