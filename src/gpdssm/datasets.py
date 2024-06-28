import numpy as np


class DataSet:

    def __init__(self, name: str, fold: int, Y: np.ndarray, X: list = None):
        self._Y = Y
        self._X = self.correct_format(X)
        self._name = name
        self._fold = fold
        self._Num_Observations, self._Dy = Y.shape

    @staticmethod
    def correct_format(X: list) -> list:
        X_new = []
        for _X in X:
            _X = np.array(_X)
            if _X.ndim == 1:
                _X = _X.reshape(-1, 1)
            X_new.append(_X)

        return X_new

    @property
    def Num_Observations(self):
        return self._Num_Observations

    @property
    def Dy(self):
        return self._Dy

    @property
    def Y(self):
        return self._Y

    @property
    def X(self):
        return self._X

    @property
    def name(self):
        return self._name

    @property
    def fold(self):
        return self._fold
