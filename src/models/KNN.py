import numpy as np
from collections import Counter

class KNN:
    def __init__(self, data, n=5):
        self.data = data
        self.n = n

    def find_nearest(self, point):
        point = np.array(point)
        X = self.data[:, :-1]  # features
        y = self.data[:, -1]   # labels
        distances = np.linalg.norm(X - point, axis=1)
        nearest_indices = np.argsort(distances)[:self.n]
        return y[nearest_indices]  # return only the labels of nearest neighbors

    def predict(self, point):
        neighbors = self.find_nearest(point)
        return Counter(neighbors).most_common(1)[0][0]
