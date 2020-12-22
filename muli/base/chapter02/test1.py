import numpy as np
a = np.array([[1, 2, 4, ([1, 2, 5])],
              [3, 2, 6, ([6, 5, 1])],
              [6, 9, 4, ([3, 7, 5])]])

print(a)
print("\n")

print(a.take(1,1))
print(a.take(1,0))
print(a.take(2))
print(a.take(1))
print(a.take(0))
print(a.take(3))

