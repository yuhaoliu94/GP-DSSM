import abc

class Likelihood:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def log_cond_prob(self, output, latent_val):
        """
        Subclass should implement log p(Y | F)
        :param output:  (size x Dout) matrix containing true outputs
        :param latent_val: (MC x size x Q) matrix of latent function values, usually Q=F
        :return:
        """
        raise NotImplementedError("Subclass should implement this.")

    @abc.abstractmethod
    def get_params(self):
        raise NotImplementedError("Subclass should implement this.")

    @abc.abstractmethod
    def predict(self, latent_val):
        raise NotImplementedError("Subclass should implement this.")

    @abc.abstractmethod
    def get_name(self, latent_val):
        raise NotImplementedError("Subclass should implement this.")

