import tensorflow as tf

from . import likelihood
import utils


class Gaussian(likelihood.Likelihood):
    # def __init__(self, log_var=-2.0):

    def log_cond_prob(self, output, mean, scale):
        return utils.log_norm_pdf(output, mean, scale)

    def get_params(self):
        return None

    def predict(self, latent_val):
        return latent_val

    def get_name(self):
        return "Regression"
