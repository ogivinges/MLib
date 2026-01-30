from collections.abc import Iterable
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import clone


class CascadeMLModel:
    def __init__(
        self,
        models,
        split_col: str,
        bins,
        n_jobs: Optional[int] = None
    ):
        self.models = models
        self.cascade_col = split_col
        self.bins = bins
        self._models = {}

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        data_cascade_categs = pd.cut(data[self.cascade_col], self.bins)
        preds = np.zeros(len(data))
        for backet in data_cascade_categs.unique():
            idx = data_cascade_categs == backet
            preds[idx.values] = self.models[backet].predict(data[idx])
        return preds

    def fit(self, X: pd.DataFrame, y):
        train_cascade_categs = pd.cut(X[self.cascade_col], self.bins)
        for itr, backet in enumerate(self.bins):
            train_idx = train_cascade_categs == backet
            if isinstance(self.models, Iterable):
                self._models[backet] = clone(self.models[itr])
            else:
                self._models[backet] = clone(self.models)
            self._models.fit(
                X[train_idx], y[train_idx]
            )
