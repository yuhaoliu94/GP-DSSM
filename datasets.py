import numpy as np

## DataSet class 
class DataSet():

    def __init__(self, Y, name, fold):
        self._num_examples = Y.shape[0]
        self._Y = Y
        self._Dout = Y.shape[1]
        self._name = name
        self._fold = fold

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def Dout(self):
        return self._Dout

    @property
    def Y(self):
        return self._Y

    @property
    def name(self):
        return self._name

    @property
    def fold(self):
        return self._fold
