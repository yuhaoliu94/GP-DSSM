import numpy as np

from abc import ABC, abstractmethod
from copy import deepcopy

from src.gpdssm.utils import get_random_state, normalize_weights, get_sequential_mse, get_sequential_mnll
from src.gpdssm.functions import RandomFeatureGP
from src.gpdssm.distributions import Normal


class Layer(ABC):

    def __init__(self, dim: int, num_particle: int = 1000, num_rff: int = 50,
                 warm_start: int = 0, learning_rate: float = 0.001, din: int = None) -> None:
        # assign after initializing structure
        self.next_layer = None
        self.prev_layer = None

        self.din = din
        self.function = None

        # scalar
        self.t = 0
        self.dim = dim
        self.dout = self.dim

        self.M = num_particle
        self.J = num_rff
        self.warm_start = warm_start
        self.learning_rate = learning_rate

    def initialize_particle_state(self) -> np.ndarray:
        normal_generator = Normal()
        return normal_generator.sample_univariate(0, 1, (self.M, self.dim))

    @abstractmethod
    def initialize_transition_function(self):
        raise NotImplementedError("Class must override initialize_transition_function")

    @abstractmethod
    def get_input_particles(self):
        raise NotImplementedError("Class must override get_input_particles")

    @abstractmethod
    def predict(self):
        raise NotImplementedError("Class must override predict")

    @abstractmethod
    def filter(self, *args):
        raise NotImplementedError("Class must override filter")

    @abstractmethod
    def update(self):
        raise NotImplementedError("Class must override update")


# Transit Layer
class TransitLayer(Layer):
    def __init__(self, *args):
        super().__init__(*args)

        self.prev_particle_state = None

        self.current_particle_state = self.initialize_particle_state()
        self.current_state = np.average(self.current_particle_state, axis=0)
        self.stored_states = [deepcopy(self.current_state)]

    def resample(self, weights: np.ndarray) -> np.ndarray:
        random_state = get_random_state()
        indices = random_state.choice(range(self.M), size=self.M, p=weights)
        return self.current_particle_state[indices, :]
        # return np.array([self.current_particle_state[m, :] for m in indices])

    def predict(self) -> None:
        predict_particle = self.function.predict(self.get_input_particles())
        self.prev_particle_state = deepcopy(self.current_particle_state)
        self.current_particle_state = predict_particle

    @abstractmethod
    def initialize_transition_function(self):
        raise NotImplementedError("Class must override initialize_transition_function")

    @abstractmethod
    def get_input_particles(self) -> np.ndarray:
        raise NotImplementedError("Class must override get_input_particles")

    @abstractmethod
    def filter(self, *args):
        raise NotImplementedError("Class must override filter")

    @abstractmethod
    def update(self):
        raise NotImplementedError("Class must override update")


class HiddenTransitLayer(TransitLayer):

    def __init__(self, *args) -> None:
        super().__init__(*args)

        self.particle_weights_for_prev_layer = np.zeros(self.M)

    def initialize_transition_function(self) -> None:
        self.din = self.dim + self.prev_layer.dim
        self.function = RandomFeatureGP(self.din, self.dout, self.J, self.warm_start, self.learning_rate)

    def get_input_particles(self) -> np.ndarray:
        input_particles = np.concatenate(
            (self.current_particle_state, self.prev_layer.current_particle_state), axis=1)

        return input_particles

    def filter(self) -> None:
        weights = self.next_layer.particle_weights_for_prev_layer

        self.current_state = np.average(self.current_particle_state, weights=weights, axis=0)
        self.stored_states.append(deepcopy(self.current_state))

        # smooth the previous state
        prev_state = np.average(self.prev_particle_state, weights=weights, axis=0)
        self.stored_states[-2] = prev_state

        replicate_current_state = np.repeat(self.current_state[np.newaxis, :], self.M, axis=0)  # M * dim
        log_likelihood = self.function.cal_log_likelihood(replicate_current_state)  # M
        self.particle_weights_for_prev_layer = normalize_weights(log_likelihood)

        self.current_particle_state = self.resample(weights)

    def update(self) -> None:
        prev_state = self.stored_states[-2]
        prev_layer_current_state = self.prev_layer.current_state
        input_vector = np.concatenate((prev_state, prev_layer_current_state))
        self.function.update(input_vector, self.current_state)

        self.t += 1


class RootTransitLayer(TransitLayer):

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def initialize_transition_function(self) -> None:
        self.din = self.dim
        self.function = RandomFeatureGP(self.din, self.dout, self.J, self.warm_start, self.learning_rate)

    def get_input_particles(self) -> np.ndarray:
        return self.current_particle_state

    def filter(self) -> None:
        weights = self.next_layer.particle_weights_for_prev_layer

        self.current_state = np.average(self.current_particle_state, weights=weights, axis=0)
        self.stored_states.append(deepcopy(self.current_state))

        # smooth the previous state
        prev_state = np.average(self.prev_particle_state, weights=weights, axis=0)
        self.stored_states[-2] = prev_state

        self.current_particle_state = self.resample(weights)

    def update(self) -> None:
        prev_state = self.stored_states[-2]
        self.function.update(prev_state, self.current_state)

        self.t += 1


class HiddenTransitInputLayer(HiddenTransitLayer):

    def initialize_transition_function(self) -> None:
        self.din += self.dim + self.prev_layer.dim
        self.function = RandomFeatureGP(self.din, self.dout, self.J, self.warm_start, self.learning_rate)

    def get_input_particles(self, u: np.ndarray = None) -> np.ndarray:
        replicate_u = np.repeat(u[np.newaxis, :], self.M, axis=0)  # M * Du
        input_particles = np.concatenate(
            (replicate_u, self.current_particle_state, self.prev_layer.current_particle_state), axis=1)

        return input_particles

    def predict(self, u: np.ndarray = None) -> None:
        predict_particle = self.function.predict(self.get_input_particles(u))
        self.prev_particle_state = deepcopy(self.current_particle_state)
        self.current_particle_state = predict_particle

    def update(self, u: np.ndarray = None) -> None:
        prev_state = self.stored_states[-2]
        prev_layer_current_state = self.prev_layer.current_state
        input_vector = np.concatenate((u, prev_state, prev_layer_current_state))
        self.function.update(input_vector, self.current_state)

        self.t += 1


class RootTransitInputLayer(RootTransitLayer):

    def initialize_transition_function(self) -> None:
        self.din += self.dim
        self.function = RandomFeatureGP(self.din, self.dout, self.J, self.warm_start, self.learning_rate)

    def get_input_particles(self, u: np.ndarray = None) -> np.ndarray:
        replicate_u = np.repeat(u[np.newaxis, :], self.M, axis=0)  # M * Du
        input_particles = np.concatenate((replicate_u, self.current_particle_state), axis=1)
        return input_particles

    def predict(self, u: np.ndarray = None) -> None:
        predict_particle = self.function.predict(self.get_input_particles(u))
        self.prev_particle_state = deepcopy(self.current_particle_state)
        self.current_particle_state = predict_particle

    def update(self, u: np.ndarray = None) -> None:
        prev_state = self.stored_states[-2]
        ensemble_input = np.concatenate((u, prev_state))
        self.function.update(ensemble_input, self.current_state)

        self.t += 1


# Non Transit Layer
class NonTransitLayer(Layer):
    def __init__(self, *args):
        super().__init__(*args)

        self.particle_weights_for_prev_layer = np.zeros(self.M)

        self.current_particle_state = None
        self.current_state = None
        self.stored_states = []

    def initialize_transition_function(self):
        self.din = self.prev_layer.dim
        self.function = RandomFeatureGP(self.din, self.dout, self.J, self.warm_start, self.learning_rate)

    def get_input_particles(self) -> np.ndarray:
        return self.prev_layer.current_particle_state

    @abstractmethod
    def predict(self):
        raise NotImplementedError("Class must override predict")

    @abstractmethod
    def filter(self, *args):
        raise NotImplementedError("Class must override filter")

    @abstractmethod
    def update(self):
        raise NotImplementedError("Class must override update")


class HiddenNonTransitLayer(NonTransitLayer):

    def predict(self) -> None:
        predict_particle = self.function.predict(self.get_input_particles())
        self.current_particle_state = predict_particle

    def filter(self) -> None:
        weights = self.next_layer.particle_weights_for_prev_layer

        self.current_state = np.average(self.current_particle_state, weights=weights, axis=0)
        self.stored_states.append(deepcopy(self.current_state))

        replicate_current_state = np.repeat(self.current_state[np.newaxis, :], self.M, axis=0)  # M * dim
        log_likelihood = self.function.cal_log_likelihood(replicate_current_state)  # M
        self.particle_weights_for_prev_layer = normalize_weights(log_likelihood)

    def update(self) -> None:
        self.function.update(self.prev_layer.current_state, self.current_state)

        self.t += 1


class HiddenNonTransitInputLayer(HiddenNonTransitLayer):

    def initialize_transition_function(self):
        self.din += self.prev_layer.dim
        self.function = RandomFeatureGP(self.din, self.dout, self.J, self.warm_start, self.learning_rate)

    def get_input_particles(self, u: np.ndarray = None) -> np.ndarray:
        replicate_u = np.repeat(u[np.newaxis, :], self.M, axis=0)  # M * Du
        input_particles = np.concatenate((replicate_u, self.prev_layer.current_particle_state), axis=1)
        return input_particles

    def predict(self, u: np.ndarray = None) -> None:
        predict_particle = self.function.predict(self.get_input_particles(u))
        self.current_particle_state = predict_particle

    def update(self, u: np.ndarray = None) -> None:
        ensemble_input = np.concatenate((u, self.prev_layer.current_state))
        self.function.update(ensemble_input, self.current_state)

        self.t += 1


class ObservationLayer(NonTransitLayer):

    def __init__(self, *args) -> None:
        super().__init__(*args)

        self.y = None
        # self.stored_y = []

        self.y_log_likelihood = 0.0
        self.stored_y_log_likelihood = []

        self.y_log_likelihood_forward = 0.0
        self.stored_y_log_likelihood_forward = []

        self.mse = 0.0
        self.mnll = 0.0

    def predict(self) -> None:
        self.current_particle_state = self.function.predict(self.get_input_particles())
        self.current_state = np.average(self.current_particle_state, axis=0)
        self.stored_states.append(deepcopy(self.current_state))

    def filter(self, y) -> None:
        self.y = y
        # self.stored_y.append(y)

        replicate_actual_realization = np.repeat(y[np.newaxis, :], self.M, axis=0)  # M * dim
        log_likelihood = self.function.cal_log_likelihood(replicate_actual_realization)  # M

        self.particle_weights_for_prev_layer = normalize_weights(log_likelihood)

        input_vector = np.average(self.prev_layer.current_particle_state, axis=0)
        cum_log_likelihood_forward = self.function.cal_y_log_likelihood_forward(input_vector, y)
        self.y_log_likelihood_forward = cum_log_likelihood_forward - np.sum(self.stored_y_log_likelihood_forward)
        self.stored_y_log_likelihood_forward.append(self.y_log_likelihood_forward)

    def update(self) -> None:
        self.function.update(self.prev_layer.current_state, self.y)

        self.t += 1

        # log_likelihood
        cum_log_likelihood = self.function.cal_y_log_likelihood()
        self.y_log_likelihood = cum_log_likelihood - np.sum(self.stored_y_log_likelihood)  # np.average(log_likelihood)
        self.stored_y_log_likelihood.append(self.y_log_likelihood)

        # mse
        self.mse = get_sequential_mse(self.mse, self.t - 1, self.y, self.current_state)

        # mnll
        self.mnll = get_sequential_mnll(self.mnll, self.t - 1, self.y_log_likelihood_forward)


class ObservationInputLayer(ObservationLayer):

    def initialize_transition_function(self):
        self.din += self.prev_layer.dim
        self.function = RandomFeatureGP(self.din, self.dout, self.J, self.warm_start, self.learning_rate)

    def get_input_particles(self, u: np.ndarray = None) -> np.ndarray:
        replicate_u = np.repeat(u[np.newaxis, :], self.M, axis=0)  # M * Du
        input_particles = np.concatenate((replicate_u, self.prev_layer.current_particle_state), axis=1)
        return input_particles

    def predict(self, u: np.ndarray = None) -> None:
        self.current_particle_state = self.function.predict(self.get_input_particles(u))
        self.current_state = np.average(self.current_particle_state, axis=0)
        self.stored_states.append(deepcopy(self.current_state))

    def filter(self, y, u: np.ndarray = None) -> None:
        self.y = y
        # self.stored_y.append(y)

        replicate_actual_realization = np.repeat(y[np.newaxis, :], self.M, axis=0)  # M * dim
        log_likelihood = self.function.cal_log_likelihood(replicate_actual_realization)  # M

        self.particle_weights_for_prev_layer = normalize_weights(log_likelihood)

        prev_input_vector = np.average(self.prev_layer.current_particle_state, axis=0)
        input_vector = np.concatenate((u, prev_input_vector))
        cum_log_likelihood_forward = self.function.cal_y_log_likelihood_forward(input_vector, y)
        self.y_log_likelihood_forward = cum_log_likelihood_forward - np.sum(self.stored_y_log_likelihood_forward)
        self.stored_y_log_likelihood_forward.append(self.y_log_likelihood_forward)

    def update(self, u: np.ndarray = None) -> None:
        ensemble_input = np.concatenate((u, self.prev_layer.current_state))
        self.function.update(ensemble_input, self.y)

        self.t += 1

        # log_likelihood
        cum_log_likelihood = self.function.cal_y_log_likelihood()
        self.y_log_likelihood = cum_log_likelihood - np.sum(self.stored_y_log_likelihood)  # np.average(log_likelihood)
        self.stored_y_log_likelihood.append(self.y_log_likelihood)

        # mse
        self.mse = get_sequential_mse(self.mse, self.t - 1, self.y, self.current_state)

        # mnll
        self.mnll = get_sequential_mnll(self.mnll, self.t - 1, self.y_log_likelihood_forward)
