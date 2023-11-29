import numpy as np


class DataBinWrapper:
    def __init__(self, max_bins):
        self.max_bins = max_bins
        self.XrangeMap = None

    def fit(self, x):
        if x.ndim == 1:
            n_features = 1
            x = x[:, np.newaxis]
        else:
            n_features = x.shape[1]
        self.XrangeMap = [[] for _ in range(n_features)]
        for idx in range(n_features):
            x_sorted = sorted(x[:, idx])
            for bin in range(1, self.max_bins):
                percent_val = np.percentile(x_sorted, (1.0 * bin / self.max_bins) * 100.0 // 1)
                self.XrangeMap[idx].append(percent_val)
            self.XrangeMap[idx] = sorted(list(set(self.XrangeMap[idx])))

    def transform(self, x):
        if x.ndim == 1:
            return np.asarray([np.digitize(x, self.XrangeMap[0])]).reshape(-1)
        else:
            return np.asarray([np.digitize(x[:, i], self.XrangeMap[i]) for i in range(x.shape[1])]).T

    def transform(self, x, XrangeMap):
        return np.asarray([np.digitize(x, XrangeMap)]).reshape(-1)


bins = DataBinWrapper(max_bins=3)
x = np.arange(30)
np.random.shuffle(x)
x = x.reshape(10, 3)

bins.fit(x)
print('0' * 100)
indexA = bins.transform(x)
print(indexA[:, 0])
print('1' * 100)
print(x)
print('2' * 100)
print(x[indexA[:, 0]])

print('3' * 100)
print(indexA)
