from typing import Literal

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

class ThresholdOptimizer:
    """Оптимизатор порога для бинарной классификации.
    
    Параметры
    ----------
    method : {"f1", "youden"}, по умолчанию "f1"
        Метод оптимизации порога:
        - "f1": максимизация F1-score
        - "youden": максимизация индекса Йодена (Youden's J statistic)
    """

    __METHODS = {"f1", "youden"}

    def __init__(self, method: Literal["f1", "youden"] = "f1"):
        if method not in self.__METHODS:
            raise ValueError(f"method {method} is not acceptable!")
        self.method = method

    def optimize(self, y_true, y_pred) -> float:
        """Находит оптимальный порог.
        
        Параметры
        ----------
        y_true : array-like формы (n_samples,)
            Истинные бинарные метки (0 или 1).
            
        y_pred : array-like формы (n_samples,)
            Предсказанные вероятности положительного класса.
            
        Возвращает
        -------
        float
            Оптимальный порог.
        """
        if self.method == "f1":
            return self._optimize_f1(y_true, y_pred)
        elif self.method == "youden":
            return self._optimize_yoden(y_true, y_pred)
        
    @staticmethod
    def _optimize_f1(y_true, y_pred) -> float:
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
        optimal_idx = np.argmax(f1_scores)
        return thresholds[optimal_idx]
    
    @staticmethod
    def _optimize_yoden(y_true, y_pred) -> float:
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        youden_index = tpr - fpr
        optimal_idx = np.argmax(youden_index)
        return thresholds[optimal_idx]