import numpy as np

from sklearn.base import BaseEstimator


class MeanModel(BaseEstimator):
    def fit(self, X, y):
        self._mean = np.mean(y)

    def predict(self, X):
        return np.full(len(X), self._mean)