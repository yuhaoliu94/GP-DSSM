import numpy as np

from abc import ABC
from scipy.stats import invgamma
from src.gpdssm.utils import get_random_state


class Distribution(ABC):

    def __init__(self):
        self.random_state = None


class Beta(Distribution):
    def __init__(self, random_state: int = 520) -> None:
        super().__init__()

        self.random_state = get_random_state(random_state)

    def sample_multivariate(self, a: np.ndarray, b: np.ndarray, size):
        return self.random_state.beta(a, b, size)

    def sample_array(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """M * C, M * C -> M * C"""
        assert a.shape == b.shape, "The shapes do not match."

        return self.random_state.beta(a, b)


class InverseGamma(Distribution):

    def __init__(self, random_state: int = 520) -> None:
        super().__init__()

        self.random_state = random_state

    def sample_array(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Dy -> Dy"""
        assert a.shape == b.shape, "The shapes do not match."

        dim = len(a)
        samples = np.zeros(dim)
        for i in range(dim):
            samples[i] = invgamma.rvs(a[i], scale=b[i], random_state=self.random_state)

        return samples


class Normal(Distribution):

    def __init__(self, random_state: int = 520) -> None:
        super().__init__()

        self.random_state = get_random_state(random_state)

    def sample_univariate(self, mean, scale, size):
        return self.random_state.normal(mean, scale, size)

    def sample_array(self, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
        """M * Dy, M * Dy -> M * Dy"""
        assert mean.shape == scale.shape, "The shapes do not match."

        return self.random_state.normal(mean, scale)
