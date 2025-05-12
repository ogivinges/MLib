import pandas as pd
import numpy as np

from typing import Callable, List, Dict
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from .metrics import mape
from .utils import is_categ, iv_local


def univariative_analysis(X: pd.DataFrame, y, model, metric: Callable = mape, is_tqdm: bool = False) -> Dict:
    res = {}
    iterator = tqdm(X.columns) if is_tqdm else X.columns
    for col in iterator:
        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=1)
        model.fit(train_x[[col]], train_y)
        preds = model.predict(test_x[[col]])
        res[col] = metric(test_y, preds)
    return res


def correlation_selection(X: pd.DataFrame, corr_threshold: float = 0.9) -> List:
    list_features = X.columns.tolist()
    correlation_matrix = X[list_features].corr()
    for itr, col in enumerate(X.columns):
        if correlation_matrix.iloc[itr, itr+1:].abs().max() > corr_threshold:
            list_features.remove(col)
    return list_features


def iv(series, target, nbins: int = 10):
    P = sum(target)
    N = series.size - P
    if is_categ(series, max_unique=nbins):
        return target.groupby(series).agg(lambda x: iv_local(x, N, P)).sum()
    return target.groupby(
        pd.qcut(series, np.linspace(0, 1, nbins+1), duplicates='drop')
    ).agg(lambda x: iv_local(x, N, P)).sum()


def backward_selection(X: pd.DataFrame, y, model, metric: Callable) -> List:
    features = X.columns.tolist()
    x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(x_train, y_train)
    if hasattr(model, "predict_proba"):
        preds = model.predict_proba(x_valid)[:, 1]
    else:
        preds = model.predict(x_valid)
    eval_metric = metric(y_valid, preds)
    for feat in X.columns:
        current_features = [c for c in features if c != feat]
        model.fit(x_train[current_features], y_train)
        if hasattr(model, "predict_proba"):
            preds = model.predict_proba(x_valid[current_features])[:, 1]
        else:
            preds = model.predict(x_valid[current_features])
        current_metric = metric(y_valid, preds)
        if current_metric >= eval_metric:
            eval_metric = current_metric
            features.remove(feat)
    return features



