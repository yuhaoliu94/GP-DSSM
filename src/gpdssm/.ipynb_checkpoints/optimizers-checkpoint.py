import numpy as np

from abc import ABC


class Optimizer(ABC):

    def __init__(self):
        self.t = 0


class AdamSGD(Optimizer):

    def __init__(self, shape: tuple, learning_rate: float = 0.1, beta_1: float = 0.9, beta_2: float = 0.99,
                 eps: float = 1.0e-8):
        super().__init__()

        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps

        self.m = np.zeros(shape)
        self.v = np.zeros(shape)

    def next_iteration(self, parameter: np.ndarray, grad: np.ndarray) -> np.ndarray:
        assert parameter.shape == grad.shape, "The shapes do not match."

        self.m = self.beta_1 * self.m + (1.0 - self.beta_1) * grad
        m_hat = self.m / (1.0 - np.power(self.beta_1, self.t + 1))

        self.v = self.beta_2 * self.v + (1.0 - self.beta_2) * np.square(grad)
        v_hat = self.v / (1.0 - np.power(self.beta_2, self.t + 1))

        delta = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)
        parameter -= delta

        self.t += 1

        return parameter
