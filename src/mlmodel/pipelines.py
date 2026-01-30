from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from ..preprocessing import LinearPreprocessor, TreePreprocessor

class SkRandomForestClassifier(Pipeline):
    def __init__(self, features, categorical_features, **kwargs):
        super().__init__(steps=[
            ('preprocessor', TreePreprocessor()),
            ('model', RandomForestClassifier(features, categorical_features, **kwargs))])
