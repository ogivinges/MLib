from collections.abc import Iterable
from typing import Literal

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import clone

from .mlmodels import MeanModel


class MLModel:
    def __init__(
            self, core: Literal["lightgbm", "mean"] = "lightgbm",
            problem_type: Literal["binary", "regression"] = "binary",
            **kwargs):
        self.core = core
        self.problem_type = problem_type
        if self.core == "lightgbm":
            if self.problem_type == "binary":
                self._model = lgb.LGBMClassifier(**kwargs)
            elif self.problem_type == "regression":
                self._model = lgb.LGBMRegressor(**kwargs)
        elif self.core == "mean":
            if self.problem_type == "regression":
                self._model = MeanModel()
            else:
                ValueError("MeanModel can be used only for regression")
        else:
            ValueError(f"Core '{self.core}' can't be used")

    def fit(
            self, train_x, train_y,
            valid_x=None, valid_y=None, eval_metric=None):
        if self.core == "lightgbm":
            self._model.fit(
                train_x, train_y,
                eval_set=[(valid_x, valid_y)], eval_metric=eval_metric)
        elif self.core == "mean":
            self._model.fit(train_x, train_y)

    def predict(self, X) -> np.ndarray:
        if self.problem_type == "binary":
            return self._model.predict_proba(X)[:, 1]
        else:
            return self._model.predict(X)


class CascadeMLModel:
    def __init__(
        self,
        models,
        split_col: str,
        bins,
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

    def __getitem__(self, key):
        return self._models[key]
