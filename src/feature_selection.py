import pandas as pd
import numpy as np

from typing import Callable
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
from .metrics import mape
from .utils import is_categ, iv_local


def univariative_analysis(X, y, model, metric: Callable = mape, is_tqdm: bool = False) -> dict:
    res = {}
    iterator = tqdm(X.columns) if is_tqdm else X.columns
    for col in iterator:
        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=1)
        model.fit(train_x[[col]], train_y)
        preds = model.predict(test_x[[col]])
        res[col] = metric(test_y, preds)
    return res


def correlation_selection(X, corr_threshold: float = 0.9) -> float:
    list_features = X.columns.tolist()
    correlation_matrix = X[list_features].corr()
    for itr, col in enumerate(X.columns):
        if correlation_matrix.iloc[itr, itr+1:].abs().max() > corr_threshold:
            list_features.remove(col)
    return list_features


def iv(series, target, nbins: int = 10) -> float:
    P = sum(target)
    N = series.size - P
    if is_categ(series, max_unique=nbins):
        return target.groupby(series).agg(lambda x: iv_local(x, N, P)).sum()
    return target.groupby(
        pd.qcut(series, np.linspace(0, 1, nbins+1), duplicates='drop')
    ).agg(lambda x: iv_local(x, N, P)).sum()



