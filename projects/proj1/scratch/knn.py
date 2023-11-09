import numpy as np
from collections import Counter

def minkowski_distance(x1, x2, p=2):
    distance = (np.sum((x1-x2)**p))**(1/p)
    return distance

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        # compute the distance
        distances = [minkowski_distance(x, x_train) for x_train in self.X_train]
    
        # get the closest k
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # majority vote
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]

    # from sklearn.metrics import accuracy_score
    # accuracy_score(y_test, y_pred)
    def accuracy_score(self, y_test, y_pred):
        acc = np.sum(y_pred == y_test) / len(y_test)
        return acc
