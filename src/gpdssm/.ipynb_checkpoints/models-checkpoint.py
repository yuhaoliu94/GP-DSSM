from abc import ABC
from copy import deepcopy
from time import time

import numpy as np

from src.gpdssm.layers import HiddenTransitLayer, RootTransitLayer, ObservationLayer, RootTransitInputLayer
from src.gpdssm.utils import import_dataset, get_mse, get_mnll, get_svd_representation_list


class StandardSingleModel(ABC):

    def __init__(self, data_name: str, data_fold: int,
                 dim_hidden: list, num_rff: int, num_particle: int,
                 display_step: int, warm_start: int, learning_rate: float, num_train_cycle: int):
        self.t = 0
        self.M = num_particle
        self.J = num_rff
        self.display_step = display_step
        self.warm_start = warm_start
        self.learning_rate = learning_rate
        self.num_train_cycle = num_train_cycle

        # data
        self.data_name = data_name
        self.data_fold = data_fold
        self.data = import_dataset(self.data_name, self.data_fold)

        # layers
        self.dim_hidden = dim_hidden  # list
        self.num_hidden_layer = len(self.dim_hidden)

        self.dim_all = self.dim_hidden + [self.data.Dy]  # list
        self.num_all_layer = self.num_hidden_layer + 1

        self.hidden_layers = []
        self.observation_layer = None
        self.layers = []
        self.functions = []
        self.initialize_structure()

        self.stored_layers = []

        self.time = []

    def initialize_structure(self, retrain=False):

        if retrain:
            self.save_current_layers()

        constant_param = (self.M, self.J, self.warm_start, self.learning_rate)

        self.hidden_layers = [RootTransitLayer(self.dim_hidden[0], *constant_param)]
        self.hidden_layers += self.construct_hidden_layers(constant_param)
        self.observation_layer = ObservationLayer(self.dim_all[-1], *constant_param)
        self.layers = self.hidden_layers + [self.observation_layer]

        for i in range(self.num_all_layer - 1):
            self.layers[i].next_layer = self.layers[i + 1]

        for i in range(1, self.num_all_layer):
            self.layers[i].prev_layer = self.layers[i - 1]

        if not retrain:
            for i in range(self.num_all_layer):
                self.layers[i].initialize_transition_function()
                self.functions.append(self.layers[i].function)
        else:
            for i in range(self.num_all_layer):
                self.layers[i].initialize_transition_function()
                self.layers[i].function = self.functions[i]

    def initialize_structure_from_model(self, model_from_different_class=None, model_from_same_class=None):
        if model_from_different_class:
            for i in range(1, self.num_all_layer):
                self.functions[i] = deepcopy(model_from_different_class.functions[i])
                self.layers[i].function = self.functions[i]

        if model_from_same_class:
            self.functions[0] = deepcopy(model_from_same_class.functions[0])
            self.layers[0].function = self.functions[0]

    def save_current_layers(self):
        self.stored_layers.append(deepcopy(self.layers))

    def construct_hidden_layers(self, constant_param):
        middle_layers = [HiddenTransitLayer(self.dim_hidden[i], *constant_param)
                         for i in range(1, self.num_hidden_layer)]
        return middle_layers

    def predict(self):
        for i in range(self.num_all_layer):
            self.layers[i].predict()

    def filter(self):
        y = self.data.Y[self.t, :]
        self.observation_layer.filter(y)
        for i in reversed(range(self.num_hidden_layer)):
            self.hidden_layers[i].filter()

    def update(self):
        for i in range(self.num_all_layer):
            self.layers[i].update()

        self.t += 1

    def learn(self) -> None:
        T = self.data.Num_Observations
        self.time.append(time())

        for n in range(self.num_train_cycle):

            print(">>> train cycle %d" % (n + 1), end="\n")

            if n:
                self.initialize_structure(retrain=True)
                self.t = 0

            for i in range(T):

                self.predict()
                self.filter()
                self.update()

                if i == 0 or i == T - 1 or (i + 1) % self.display_step == 0:
                    self.log_print(n + 1)

    def relearn(self) -> None:
        T = self.data.Num_Observations
        self.time.append(time())

        print(">>> relearn", end="\n")

        self.initialize_structure(retrain=True)
        self.t = 0

        for i in range(T):

            self.predict()
            self.filter()
            self.update()

            if i == 0 or i == T - 1 or (i + 1) % self.display_step == 0:
                self.log_print(1)

    def log_print(self, cycle: int) -> None:
        cum_mse = np.round(self.observation_layer.mse, 4)
        cum_mnll = np.round(self.observation_layer.mnll, 4)

        current_time = time()
        self.time.append(current_time)
        minutes = np.round((current_time - self.time[0]) / 60, 2)

        print(">>> cycle=%s, t=%s, mse=%s, mnll=%s, time=%s minutes" %
              (cycle, self.t, cum_mse, cum_mnll, minutes), end="\n")

    def get_observations(self) -> np.ndarray:
        return self.data.Y

    def get_predictions(self) -> np.ndarray:
        return np.array(self.observation_layer.stored_states)

    def get_actual_states(self) -> list:
        return self.data.X

    def get_estimated_states(self) -> list:
        return [np.array(self.hidden_layers[i].stored_states) for i in range(self.num_hidden_layer)]

    def get_log_likelihoods(self) -> np.ndarray:
        return np.array(self.observation_layer.stored_y_log_likelihood)

    def get_log_likelihoods_forward(self) -> np.ndarray:
        return np.array(self.observation_layer.stored_y_log_likelihood_forward)

    def get_mse(self) -> np.ndarray:
        return get_mse(self.data.Y, self.get_predictions())

    def get_mnll(self) -> np.ndarray:
        return get_mnll(np.array(self.observation_layer.stored_y_log_likelihood))

    def get_svd_representation(self, last_num_sample: int = None) -> dict:
        actual_states = self.get_actual_states()
        if actual_states:
            actual_svd_representation = get_svd_representation_list(actual_states, last_num_sample)
        else:
            actual_svd_representation = None

        estimated_svd_representation = get_svd_representation_list(self.get_estimated_states(), last_num_sample)

        return {"actual_svd_representation": actual_svd_representation,
                "estimated_svd_representation": estimated_svd_representation}


class InputSingleModel(StandardSingleModel):
    def initialize_structure(self, retrain=False):

        if retrain:
            self.save_current_layers()

        constant_param = (self.M, self.J, self.warm_start, self.learning_rate)

        self.hidden_layers = [RootTransitInputLayer(self.dim_hidden[0], *constant_param, self.data.Du)]
        self.hidden_layers += self.construct_hidden_layers(constant_param)
        self.observation_layer = ObservationLayer(self.dim_all[-1], *constant_param)
        self.layers = self.hidden_layers + [self.observation_layer]

        for i in range(self.num_all_layer - 1):
            self.layers[i].next_layer = self.layers[i + 1]

        for i in range(1, self.num_all_layer):
            self.layers[i].prev_layer = self.layers[i - 1]

        if not retrain:
            for i in range(self.num_all_layer):
                self.layers[i].initialize_transition_function()
                self.functions.append(self.layers[i].function)
        else:
            for i in range(self.num_all_layer):
                self.layers[i].initialize_transition_function()
                self.layers[i].function = self.functions[i]

    def predict(self):
        u = self.data.U[self.t, :]
        self.layers[0].predict(u)
        for i in range(1, self.num_all_layer):
            self.layers[i].predict()

    def update(self):
        u = self.data.U[self.t, :]
        self.layers[0].update(u)
        for i in range(1, self.num_all_layer):
            self.layers[i].update()

        self.t += 1