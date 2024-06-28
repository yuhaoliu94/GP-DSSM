from abc import ABC, abstractmethod
import numpy as np
from scipy.special import loggamma


class Likelihood(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def get_loglikelihood(self):
        raise NotImplementedError("Class must override get_loglikelihood")


class MarginalLikelihood(Likelihood):

    def __init__(self, t: int, det_Lambda: float, a0: np.ndarray, a: np.ndarray, b0: np.ndarray, b: np.ndarray):
        super().__init__()
        assert det_Lambda > 0, "det_Lambda is less than 0."

        self.t = t
        self.det_Lambda = det_Lambda
        self.a0, self.a, self.b0, self.b = a0, a, b0, b

    def get_loglikelihood(self) -> np.ndarray:
        assert (self.b > 0).all(), "b is less than 0"

        log_likelihood = loggamma(self.a) - loggamma(self.a0)
        log_likelihood += self.a0 * np.log(self.b0) - self.a * np.log(self.b)
        log_likelihood -= 0.5 * self.t * np.log(2 * np.pi) + 0.5 * np.log(self.det_Lambda)

        return log_likelihood


class NormalLikelihood(Likelihood):
    def __init__(self, y: np.ndarray, mean: np.ndarray, scale: np.ndarray):
        super().__init__()

        self.y = y
        self.mean = mean
        self.scale = scale

    def get_loglikelihood(self) -> np.ndarray:
        var = np.square(self.scale)
        log_likelihood = np.log(2 * np.pi * var) + np.square(self.y - self.mean) / var
        log_likelihood *= -0.5

        return log_likelihood
