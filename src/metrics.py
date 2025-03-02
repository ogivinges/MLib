from sklearn.metrics import mean_absolute_percentage_error as mape


def mape_special(y_true, preds, threshold: float = 0) -> float:
    return mape(y_true[y_true > threshold], preds[y_true > threshold])