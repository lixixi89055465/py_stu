import numpy as np

# a = [1, 2]
# b = [3, 4]
# c = np.vstack((a, b))
# print(c)
# a = [1, 2]
# b = [3, 4]
# c = np.vstack((a, b))
# print(c)
# print('#' * 100)
# print(np.hstack([[1, 2], [3]]))
# print('#' * 100)
# a = [[1, 2],
#      [3, 4]]
# b = [[5],
#      [6]]
# print(np.hstack([a, b]))

# a = np.arange(120).reshape(2, 3, 4, 5)
# b = np.arange(120, 160).reshape(2, 1, 4, 5)
# size = a.shape[1]
# c = np.hstack([a, b])
# print(np.sum(c[:, :size] - a))
# print(np.sum(c[:, size:] - b))

# a = np.array([1, 2])
# b = np.array([3, 4])
# print(np.dstack((a, b)))
# a = [
#     [0, 3],
#     [1, 4],
#     [2, 5],
# ]
# b = [
#     [6, 9],
#     [7, 10],
#     [8, 11],
# ]
# c = np.dstack([a, b])
# print(c)
a = np.arange(120).reshape(2, 3, 2, 2, 5)
b = np.arange(120, 240).reshape(2, 3, 2, 2, 5)
c = np.dstack([a, b, b])
print(np.sum(c[:,:,:2,:,:]))
print(np.sum(c[:,:,2:4,:,:]))
print(np.sum(c[:,:,4:6,:,:]))
