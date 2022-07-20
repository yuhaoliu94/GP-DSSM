from . import likelihood
import utils

class Student_t(likelihood.Likelihood):
    # def __init__(self):

    def log_cond_prob(self, output, mean, scale, nu):
        return utils.log_t_pdf(output, mean, scale, nu)

    def get_params(self):
        return None

    def predict(self, latent_val):
        return latent_val

    def get_name(self):
        return "Regression"
