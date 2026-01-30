from typing import Protocol, Any

import numpy as np

class MLModelProtocol(Protocol):
    def fit(self, X, y, **kwargs) -> 'MLModelProtocol': ...

    def predict(self, X: Any) -> np.ndarray: ...
