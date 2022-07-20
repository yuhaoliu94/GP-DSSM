import numpy as np
import tensorflow as tf
from . import likelihood
import utils

class Softmax(likelihood.Likelihood):
    """
    Implements softmax likelihood for multi-class classification
    """
#    def __init__(self):

    def log_cond_prob(self, output, latent_val):
        return np.sum(output * latent_val, 1) - utils.logsumexp(latent_val, 1)

    def predict(self, latent_val):
        """
        return the probabilty for all the samples, datapoints and calsses
        :param latent_val:
        :return:
        """
        logprob = latent_val - np.expand_dims(utils.logsumexp(latent_val, 1), 1)
        return np.exp(logprob)

    def get_params(self):
        return None

    def get_name(self):
        return "Classification"
