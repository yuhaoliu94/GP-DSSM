import os
import pickle

import numpy as np

from src.gpdssm.datasets import DataSet


def onehot_array(_array: np.ndarray, num_class) -> np.ndarray:
    onehot_matrix = np.zeros((_array.size, num_class))
    onehot_matrix[np.arange(_array.size), _array] = 1
    return onehot_matrix


def import_dataset(data_name: str, data_fold: int) -> DataSet:
    cwd = os.getcwd()
    Y_path = os.path.join(cwd, "folds", "%s_fold_%s_Y.txt" % (data_name, data_fold))
    Y = np.loadtxt(Y_path, delimiter=' ')

    if Y.ndim == 1:
        Y = np.reshape(Y, (-1, 1))

    X_path = os.path.join(cwd, "folds", "%s_fold_%s_X.pickle" % (data_name, data_fold))
    if os.path.exists(X_path):
        with open(X_path, 'rb') as f:
            X = pickle.load(f)
        data = DataSet(data_name, data_fold, Y, X)
    else:
        data = DataSet(data_name, data_fold, Y)

    return data


def symmetric_mult(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    C = np.matmul(A, B)
    D = np.matmul(C, A.T)
    return D


def get_random_state(random_state: int = 520):
    return np.random.RandomState(random_state)


def normalize_weights(log_likelihood: np.ndarray):
    log_likelihood -= np.max(log_likelihood)
    weights = np.exp(log_likelihood)
    weights /= np.sum(weights)
    return weights


def normalize_weights_special(log_likelihood: np.ndarray):
    idx = log_likelihood != 0.0
    log_likelihood[idx] -= np.max(log_likelihood[idx])
    log_likelihood[idx] = np.exp(log_likelihood[idx])
    log_likelihood[idx] /= np.sum(log_likelihood[idx])
    return log_likelihood


def normalize_2d(array: np.ndarray) -> np.ndarray:
    mean = np.mean(array, axis=0)
    std = np.std(array, axis=0)
    return (array - mean) / std


def get_svd_representation(matrix: np.ndarray) -> np.ndarray:
    _, dim = matrix.shape
    U, _, _ = np.linalg.svd(matrix)
    return U[:, :dim]


def get_svd_representation_list(matrix_list: list, last_num_sample: int) -> list:
    num_matrix = len(matrix_list)
    svd_matrix_list = []
    for i in range(num_matrix):
        sub_matrix = matrix_list[i][-last_num_sample:, :] if last_num_sample else matrix_list[i]
        svd_matrix = get_svd_representation(sub_matrix)
        svd_matrix_list.append(svd_matrix)

    return svd_matrix_list


def get_mse(observations: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    se = np.sum(np.square(observations - predictions), axis=1)
    return np.cumsum(se) / np.arange(1, len(se) + 1)


def get_sequential_mse(prev_mse: float, num_observation: int, observation: np.ndarray, prediction: np.ndarray) -> float:
    se = prev_mse * num_observation
    se += np.sum(np.square(observation - prediction))
    return se / (num_observation + 1)


def get_mnll(log_likelihood: np.ndarray) -> np.ndarray:
    nll = -log_likelihood
    return np.cumsum(nll) / np.arange(1, len(nll) + 1)


def get_sequential_mnll(prev_mnll: float, num_observation: int, log_likelihood: float) -> float:
    nll = prev_mnll * num_observation
    nll -= log_likelihood
    return nll / (num_observation + 1)
