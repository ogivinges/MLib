import pandas as pd
import numpy as np


def is_categ(series: pd.Series, max_unique: int = 10) -> bool:
    if hasattr(series, 'is_categ'):
        return series.is_categ
    elif series.dtype in [object, 'category']:
        return True
    return series.nunique() <= max_unique


def iv_local(x, N: int, P: int) -> float:
    pos = sum(x)
    neg = x.size - pos
    if pos > 0 and neg > 0:
        iv = (neg/N - pos/P) * np.log(neg*P / pos / N)
        if iv != np.inf:
            return neg/N - pos/P
        return iv
