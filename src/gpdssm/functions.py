import numpy as np

from abc import ABC

from src.gpdssm.utils import symmetric_mult
from src.gpdssm.optimizers import AdamSGD
from src.gpdssm.likelihoods import NormalLikelihood, MarginalLikelihood, BinaryLikelihood
from src.gpdssm.distributions import InverseGamma, Normal


class Function(ABC):

    def __init__(self):
        self.t = 0


class RandomFeatureGP(Function):

    def __init__(self, din: int, dout: int, num_rff: int, warm_start: int, learning_rate: float) -> None:
        super().__init__()

        # int scalar
        self.J = num_rff
        self.din = din
        self.dmid = 2 * self.J
        self.dout = dout
        self.warm_start = warm_start
        self.learning_rate = learning_rate

        self.v0 = np.ones(self.dout) * 6.0
        self.s0 = np.ones(self.dout) * 0.2
        self.a0 = self.v0 / 2
        self.b0 = 0.5 * self.v0 * np.power(self.s0, 2)

        self.a = self.v0 / 2
        self.b = 0.5 * self.v0 * np.power(self.s0, 2)
        self.p = np.zeros(self.dout) * 0.0  # yTy -> 1 -> Dy
        self.Q = np.zeros((self.dmid, self.dout))  # phi(X)TY -> 2J * 1 -> 2J * Dy

        self.inverse_gamma_generator = InverseGamma()
        self.normal_generator = Normal()

        self.Omega = self.initialize_Omega()  # Dx * J
        self.Theta = np.zeros((self.dmid, self.dout))  # 2J * Dy
        self.Lambda = np.eye(self.dmid)  # 2J * 2J #np.repeat(tmp_matrix[np.newaxis, :, :], self.dout, axis=0)
        self.Lambda_inverse = np.eye(self.dmid)  # 2J * 2J #np.repeat(tmp_matrix[np.newaxis, :, :], self.dout, axis=0)

        self.optimizer = AdamSGD(self.Omega.shape, learning_rate=self.learning_rate)

        self.predict_dist = [None, None]

    def initialize_Omega(self):
        return self.normal_generator.sample_array(np.zeros((self.din, self.J)), np.ones((self.din, self.J)))

    def predict(self, input_particles: np.ndarray) -> np.ndarray:
        """M * Dx --> M * Dy"""

        mul_matrix = np.matmul(input_particles, self.Omega)  # M * J
        trigo_matrix = np.concatenate((np.sin(mul_matrix), np.cos(mul_matrix)), axis=1) / np.sqrt(self.J)  # M * 2J
        output_particles_mean = np.matmul(trigo_matrix, self.Theta)  # M * Dy

        sigma2 = self.inverse_gamma_generator.sample_array(self.a, self.b)  # Dy
        var_sym = symmetric_mult(trigo_matrix, self.Lambda_inverse)  # M * M
        var_diag = (np.diag(var_sym) + 1)  # M
        var = np.repeat(var_diag[:, np.newaxis], len(sigma2), axis=1) * sigma2
        scale = np.sqrt(var)

        output_particles = self.normal_generator.sample_array(output_particles_mean, scale)  # M * Dy

        self.predict_dist = [output_particles_mean, scale]

        return output_particles

    def update(self, input_vector: np.ndarray, output_vector: np.ndarray) -> None:
        """Dx, Dy"""

        mul_vector = np.matmul(input_vector, self.Omega)  # J
        trigo_vector = np.concatenate((np.sin(mul_vector), np.cos(mul_vector))) / np.sqrt(self.J)  # 2J

        self.Lambda += np.matmul(trigo_vector.reshape(-1, 1), trigo_vector.reshape(1, -1))  # 2J * 2J
        self.Lambda_inverse = np.linalg.inv(self.Lambda + np.eye(self.dmid) * 1.0e-8)  # 2J * 2J

        self.p += np.power(output_vector, 2)  # Dy
        self.Q += np.repeat(trigo_vector[:, np.newaxis], len(output_vector), axis=1) * output_vector  # 2J * Dy
        self.Theta = np.matmul(self.Lambda_inverse, self.Q)  # 2J * Dy

        self.a += 0.5  # Dy
        self.b = self.b0 + 0.5 * (self.p - np.diag(np.matmul(self.Q.T, self.Theta)))  # Dy

        if self.t >= self.warm_start:
            grad_Omega = self.cal_grad_Omega(input_vector, output_vector)
            self.Omega = self.optimizer.next_iteration(self.Omega, grad_Omega)

        self.t += 1

    # MSE loss
    def cal_grad_Omega(self, input_vector: np.ndarray, output_vector: np.ndarray) -> np.ndarray:
        """Dx, Dy -> Dx * J"""

        mul_vector = np.matmul(input_vector, self.Omega)  # J
        trigo_vector = np.concatenate((np.sin(mul_vector), np.cos(mul_vector))) / np.sqrt(self.J)  # 2J
        predict_mean = np.matmul(trigo_vector, self.Theta)  # Dy

        part_1 = np.diag(np.cos(mul_vector))  # J * J
        part_2 = np.diag(-np.sin(mul_vector))  # J * J
        trigo_matrix = np.concatenate((part_1, part_2), axis=1) / np.sqrt(self.J)  # J * 2J

        tmp_matrix = np.matmul(trigo_matrix, self.Theta).T  # Dy * J
        grad_matrix_raw = np.repeat(tmp_matrix[:, :, np.newaxis], len(input_vector),
                                    axis=2) * input_vector  # Dy * J * Dx
        grad_matrix_weighted = grad_matrix_raw.T * (predict_mean - output_vector)  # Dx * J * Dy
        grad_matrix = 2 * np.sum(grad_matrix_weighted, axis=2)  # Dx * J

        return grad_matrix

    def cal_log_likelihood(self, y: np.ndarray):
        """M * Dy"""
        marginal_dist = NormalLikelihood(y, *self.predict_dist)
        log_likelihood = marginal_dist.get_loglikelihood()
        return np.sum(log_likelihood, axis=1)

    def cal_y_log_likelihood(self) -> float:
        det_Lambda = np.linalg.det(self.Lambda)
        marginal_dist = MarginalLikelihood(self.t, det_Lambda, self.a0, self.a, self.b0, self.b)
        log_likelihood = marginal_dist.get_loglikelihood()
        return np.sum(log_likelihood)

    def cal_y_log_likelihood_forward(self, input_vector: np.ndarray, output_vector: np.ndarray) -> float:

        # fake update
        mul_vector = np.matmul(input_vector, self.Omega)  # J
        trigo_vector = np.concatenate((np.sin(mul_vector), np.cos(mul_vector))) / np.sqrt(self.J)  # 2J

        Lambda = self.Lambda + np.matmul(trigo_vector.reshape(-1, 1), trigo_vector.reshape(1, -1))  # 2J * 2J
        Lambda_inverse = np.linalg.inv(Lambda + np.eye(self.dmid) * 1.0e-8)  # 2J * 2J

        p = self.p + np.power(output_vector, 2)  # Dy
        Q = self.Q + np.repeat(trigo_vector[:, np.newaxis], len(output_vector), axis=1) * output_vector  # 2J * Dy
        Theta = np.matmul(Lambda_inverse, Q)  # 2J * Dy

        a = self.a + 0.5  # Dy
        b = self.b0 + 0.5 * (p - np.diag(np.matmul(Q.T, Theta)))  # Dy

        # cal_y_log_likelihood
        det_Lambda = np.linalg.det(Lambda)
        marginal_dist = MarginalLikelihood(self.t + 1, det_Lambda, self.a0, a, self.b0, b)
        log_likelihood = marginal_dist.get_loglikelihood()
        return np.sum(log_likelihood)


class Sigmoid(Function):
    def __init__(self, din: int, dout: int, warm_start: int, learning_rate: float) -> None:
        super().__init__()

        # int scalar
        self.din = din
        self.dout = dout

        assert self.dout == self.din, "The dimension for Sigmoid is wrong."
        self.warm_start = warm_start
        self.learning_rate = learning_rate

        self.alpha = np.zeros(self.dout)
        self.beta = -np.ones(self.dout)

        self.optimizer_alpha = AdamSGD((self.dout,), learning_rate=self.learning_rate)
        self.optimizer_beta = AdamSGD((self.dout,), learning_rate=self.learning_rate)

        self.predict_dist = None

        self.y_log_likelihood = []

    def get_sigmoid(self, input_array: np.ndarray):
        output_array = 1 / (1 + np.exp(self.alpha + self.beta * input_array))

        return output_array

    def cal_grad(self, input_vector: np.ndarray, output_vector: np.ndarray) -> (np.ndarray, np.ndarray):
        """Dy, Dy -> Dy"""
        sigmoid = self.get_sigmoid(input_vector)
        part1 = 0. - output_vector * (1. - sigmoid)
        part2 = (1. - output_vector) * sigmoid
        grad_alpha = - (part1 + part2)
        grad_beta = grad_alpha * input_vector

        return grad_alpha, grad_beta

    def predict(self, input_particles: np.ndarray) -> np.ndarray:
        """M * Dy --> M * Dy"""
        output_particles = self.get_sigmoid(input_particles)

        self.predict_dist = output_particles
        return output_particles

    def update(self, input_vector: np.ndarray, output_vector: np.ndarray) -> None:
        """Dy, Dy"""
        if self.t >= self.warm_start:
            grad_alpha, grad_beta = self.cal_grad(input_vector, output_vector)
            self.alpha = self.optimizer_alpha.next_iteration(self.alpha, grad_alpha)
            self.beta = self.optimizer_beta.next_iteration(self.beta, grad_beta)

        self.t += 1

    def cal_log_likelihood(self, y: np.ndarray):
        """M * Dy"""
        marginal_dist = BinaryLikelihood(y, self.predict_dist)
        log_likelihood = marginal_dist.get_loglikelihood()
        cumdim_log_likelihood = np.sum(log_likelihood, axis=1)
        self.y_log_likelihood.append(np.average(cumdim_log_likelihood))
        return cumdim_log_likelihood

    def cal_y_log_likelihood(self) -> float:
        return np.sum(self.y_log_likelihood)

    def cal_y_log_likelihood_forward(self, input_vector: np.ndarray, output_vector: np.ndarray) -> float:
        output_vector_hat = self.get_sigmoid(input_vector)
        marginal_dist = BinaryLikelihood(output_vector, output_vector_hat)
        log_likelihood = marginal_dist.get_loglikelihood()
        cumdim_log_likelihood = np.sum(log_likelihood)
        return cumdim_log_likelihood + np.sum(self.y_log_likelihood[:-1])