import numpy as np


class DataSet:

    def __init__(self, name: str, fold: int, Y: np.ndarray, X: list = None, U: np.ndarray = None):
        self._Y = self.correct_shape(Y)
        self._U = self.correct_shape(U)
        self._X = self.correct_format(X)
        self._name = name
        self._fold = fold
        self._Num_Observations, self._Dy = self.Y.shape
        if U is not None:
            self._Num_Observations_U, self._Du = self._U.shape
            assert self.Num_Observations == self._Num_Observations_U, ("The sample sizes are not consistent for input "
                                                                       "and output.")
        else:
            self._Num_Observations_U, self._Du = None, None

    @staticmethod
    def correct_format(X: list) -> list:
        X_new = []
        for _X in X:
            _X = np.array(_X)
            if _X.ndim == 1:
                _X = _X.reshape(-1, 1)
            X_new.append(_X)

        return X_new

    @staticmethod
    def correct_shape(_X: np.ndarray) -> np.ndarray:
        if _X.ndim == 1:
            _X = np.reshape(_X, (-1, 1))

        return _X

    @property
    def Num_Observations(self):
        return self._Num_Observations

    @property
    def Dy(self):
        return self._Dy

    @property
    def Du(self):
        return self._Du

    @property
    def Y(self):
        return self._Y

    @property
    def X(self):
        return self._X

    @property
    def U(self):
        return self._U

    @property
    def name(self):
        return self._name

    @property
    def fold(self):
        return self._fold
