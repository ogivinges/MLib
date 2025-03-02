import optuna
import numpy as np

from typing import Callable
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from ..src.metrics import mape


def model_bootstrap(X, y, model, metric: Callable = mape, N: int = 100, is_tqdm: bool = False) -> list:
    result = []
    iterator = tqdm(range(N)) if is_tqdm else range(N)
    for i in iterator:
        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2)
        model.fit(train_x, train_y)
        preds = model.predict(test_x)
        result.append(metric(test_y, preds))
    return result


# Конвертация словаря с гиперпараметрами в словарь для optuna
def parse_optuna_params(trial: optuna.Trial, params: dict) -> dict:
    optuna_params = {}
    for param in params:
        if not isinstance(params[param], dict):
            optuna_params[param] = param[param]
        else:
            _params = params[param].copy()
            param_type = _params.pop('type')
            if param_type == int:
                optuna_params[param] = trial.suggest_int(param, **_params)
            elif param_type == float:
                optuna_params[param] = trial.suggest_float(param, **_params)
    return optuna_params


def objective(trial: optuna.Trial, X, y, params: dict, core_model) -> float:
    params = parse_optuna_params(trial, params)
    model = core_model(**params)
    result = model_bootstrap(X, y, model=model, N=100, is_tqdm=False)
    return np.mean(result)
                

