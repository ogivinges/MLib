import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

class CategTargetEncoding(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        
        self.encoding = {}
        self.global_mean = np.mean(y)

        tmp = pd.concat([X, y], axis=1)
        for col in X.columns:
            self.encoding[col] = tmp.groupby(col)[y.name].mean().to_dict()
        return self
    
    def transform(self, X):
        check_is_fitted(self, 'encoding')
        return X.apply(
            lambda col: col.map(self.encoding[col.name]).fillna(self.global_mean))
    

class CategoricalFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, max_unique=3):
        self.max_unique = max_unique
        self.categorical_features = []
        self.categories = {}

    def fit(self, X, y=None):
        for feature in X:
            if self.is_categ(X[feature], max_unique=self.max_unique):
                self.categorical_features.append(feature)
                self.categories[feature] = X[feature].fillna("nan").astype(str).astype('category').cat.categories
        return self
    
    def transform(self, X: pd.DataFrame):
        for feature in self.categorical_features:
            if feature in X.columns:
                X[feature] = pd.Categorical(X[feature].fillna("nan").astype(str), categories=self.categories[feature])
                X[feature].is_category = True
        return X
    
    @staticmethod
    def is_categ(series: pd.Series, max_unique: int = 3) -> bool:
        if hasattr(series, 'is_categ'):
            return series.is_categ
        elif series.dtype in [object, 'category']:
            return True
        return series.nunique() <= max_unique
    
    def save(self, file_name):
        joblib.dump(self, file_name)
    

class PSICalculator():
    def __init__(self, bins=10, epsilon=1e-6):
        self.bins = bins
        self.epsilon = epsilon

    def fit(self, X, y=None):
        self.bins_info_ = {}
        self.expected_proportion_ = {}
        for feature in X:
            if CategoricalFeatures.is_categ(X[feature]):
                self._fit_categorical(X[feature])
            else:
                self._fit_continuous(X[feature])
        return self
    
    def _fit_categorical(self, series):
        categories = series.unique()
        counts = series.value_counts().reindex(categories, fill_values=0)
        counts = counts + self.epsilon
        counts['other'] = self.epsilon

        self.bins_info_[series.name] = {
            'type': 'categorical',
            'categories': categories.tolist()
        }

        total = counts.sum()
        self.expected_proportion_[series.name] = counts / total

    def _fit_continuous(self, series):
        bin_series, edges = pd.qcut(
            series, np.linspace(0, 1, self.bins+1),
            retbins=True, duplicates='drop')
        self.bins_info_[series.name] = {
            'type': 'continuous',
            'edges': [-np.inf] + edges[1:-1].tolist() + [np.inf]
        }
        self.expected_proportion_[series.name] = bin_series.value_counts() / len(series)

    def calculate(self, X):
        psi_results = {}
        for feature in X:
            if feature in self.bins_info_.kesy():
                if self.bins_info_[feature]['type'] == 'categorical':
                    psi = self._calculate_categorical_psi(X[feature], feature)
                else:
                    psi = self._calculate_continuous_psi(X[feature], feature)
                psi_results[feature] = psi
        return psi_results
    
    def _calculate_categorical_psi(self, series, feature):
        categories = self.bins_info_[feature]['categories']
        mapped_series = series.where(series.isin(categories), 'other')

        counts = mapped_series.value_counts().reindex(
            categories + ['other'], fill_value=0)
        counts = counts + self.epsilon

        actual_proportions = counts / counts.sum()
        expected_proportions = self.expected_proportion_[feature]
        return self._compute_psi(expected_proportions, actual_proportions)
    
    def _calculate_continuous_psi(self, series, feature):
        counts = pd.cut(series, self.bins_info_[feature]['edges'], right=False).value_counts()
        counts = counts + self.epsilon

        actual_proportions = counts / counts.sum()
        expected_proportions = self.expected_proportion_[feature]
        return self._compute_psi(expected_proportions, actual_proportions)
    
    @staticmethod
    def _compute_psi(expected,  actual):
        return np.sum((expected - actual) * np.log(expected / actual))
    
    def save(self, file_name):
        joblib.dump(self, file_name)


