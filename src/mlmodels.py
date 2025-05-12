import numpy as np
import pandas as pd

from autogluon.tabular import TabularPredictor
from autowoe import AutoWoE
from sklearn.base import BaseEstimator


class MeanModel(BaseEstimator):
    def fit(self, X, y):
        self._mean = np.mean(y)

    def predict(self, X):
        return np.full(len(X), self._mean)


class AutoGluonClassifier(BaseEstimator):
    def __init__(self, time_limit=120, eval_metric='roc_auc', presets='best_quality', verbose=-1):
        self.time_limit = time_limit
        self.eval_metric = eval_metric
        self.presets = presets
        self.predictor = None
        self.verbose = verbose

        
    def fit(self, X, y):
        data = X.copy()
        data['target'] = y
        self.predictor = TabularPredictor(
            label='target',
            eval_metric=self.eval_metric,
            problem_type='binary',
            verbosity=self.verbose
        ).fit(
            train_data=data,
            time_limit=self.time_limit,
            presets=self.presets
        )
        return self
        
    def predict(self, X):
        return self.predictor.predict(X).values
        
    def predict_proba(self, X):
        return self.predictor.predict_proba(X).values
        
    def score(self, X, y):
        return self.predictor.evaluate(pd.DataFrame(X).assign(target=y))[self.eval_metric]
    

# Работает паршиво, хорошо бы вообще это с нуля переписать
class AutoWoeClassifier(BaseEstimator):
    def __init__(self, th_nan=0.0, th_cat=0.0, n_jobs=1, verbose=2):
        self.th_nan = th_nan
        self.th_cat = th_cat
        self.n_jobs = n_jobs
        self.verbose = verbose
        self._model = AutoWoE(
            task='BIN', 
            th_nan=self.th_nan,
            th_cat=self.th_cat,
            n_jobs=self.n_jobs, 
            verbose=self.verbose)

    def fit(self, X: pd.DataFrame, y=None):
        self._model.fit(
            pd.concat([X, y], axis=1), 
            target_name=y.name, 
            features_type={key: ('cat' if val=='category' else 'real') for key, val in X.dtypes.to_dict().items()}
        )
        return self

    def predict_proba(self, X: pd.DataFrame):
        return self._model.predict_proba(X)

    def predict(self, X: pd.DataFrame):
        return self._model.predict(X)