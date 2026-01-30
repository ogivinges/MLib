from typing import List

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .preprocessing import CategTargetEncoding


class LinearPreprocessor(ColumnTransformer):
    def __init__(self, features: List, categorical_features: List):
        super().__init__(
            transformers=[
                ('cat', Pipeline([
                    ('target_encoding', CategTargetEncoding())
                ]), categorical_features),
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', StandardScaler())
                ]), [c for c in features if c not in categorical_features])
            ]
        )


class TreePreprocessor(ColumnTransformer):
    def __init__(self, features: List, categorical_features: List):
        super().__init__(
            transformers=[
                ('cat', Pipeline([
                    ('target_encoding', CategTargetEncoding())
                ]), categorical_features),
                ('num', Pipeline([
                    ('imputer', SimpleImputer(
                        fill_value=-999, strategy='constant'))
                ]), [c for c in features if c not in categorical_features])
            ]
        )
