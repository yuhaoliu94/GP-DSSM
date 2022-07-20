from __future__ import print_function

import numpy as np
import matplotlib

matplotlib.use('Agg')
from collections import deque
import utils
import likelihoods
import random


class dgpssm(object):
    def __init__(self, init_dataset, d_out, n_layers, n_rff, df, M, kernel_type, prior_var, index):
        """
        :param d_out: Dimensionality of the output
        :param n_layers: Number of hidden layers
        :param n_rff: Number of random features for each layer
        :param df: Number of GPs for each layer
        :param kernel_type: Kernel type: currently only random Fourier features for RBF and arccosine kernels are implemented
        :param index: The index of candidate model
        :param break_dim: Which part is regression or classification
        """
        self.reg_likelihood = likelihoods.Student_t()
        self.prior_likelihood = likelihoods.Gaussian()
        self.kernel_type = kernel_type
        self.init_dataset = init_dataset

        ## These are all scalars
        self.nl = n_layers  ## Number of layers
        self.n_Omega = n_layers  ## Number of weight matrices is "Number of layers"
        self.n_W = n_layers
        self.index = index
        self.M = M
        self.prior_var = prior_var
            
        ## These are arrays to allow flexibility in the future
        self.n_rff = n_rff * np.ones(n_layers, dtype=np.int32)
        self.df = df * np.ones(n_layers - 1, dtype=np.int32) # df * np.ones(n_layers - 1, dtype=np.int32)
        self.d_x = df[0]

        ## Dimensionality of Omega matrices
        self.d_in = np.concatenate([[self.d_x] * 2, self.df[1:]])
        self.d_out = self.n_rff

        ## Dimensionality of W matrices
        if self.kernel_type == "RBF":
            self.dhat_in = self.n_rff * 2
            self.dhat_out = np.concatenate([self.df, [d_out]])

        if self.kernel_type == "arccosine":
            self.dhat_in = self.n_rff
            self.dhat_out = np.concatenate([self.df, [d_out]])

        self.T = 0

        ## Initialize posterior parameters
        self.log_theta_sigma2 = self.init_prior_log_theta_sigma2()

        self.Omega = self.init_prior_Omega()

        self.mean_W, self.Sigma_W = self.init_prior_W()

        self.Xs = self.init_prior_X0()
        self.Xhat = [[np.zeros(self.d_x)]]

        self.Phi = self.init_prior_Phi()

        self.r = self.init_prior_r()

    ## Function to initialize the posterior over Omega
    def init_prior_Omega(self):
        Omega = []
        for i in range(self.n_Omega):
            Lambda = 1 / utils.get_random(size=self.d_in[i])
            Omega_Naive = utils.get_mvn_samples(mean=np.zeros(self.d_in[i]), cov=np.diag(Lambda), shape=self.d_out[i])
            Omega.append(np.atleast_2d(Omega_Naive).T)

        return Omega

    ## Function to initialize the posterior over W
    def init_prior_W(self):
        mean_W = [np.zeros([self.dhat_out[i], self.dhat_in[i]]) for i in range(self.n_W)]
        Sigma_W = [np.zeros([self.dhat_out[i], self.dhat_in[i], self.dhat_in[i]]) for i in range(self.n_W)]
        for i in range(self.n_W):
            for j in range(self.dhat_out[i]):
                Sigma_W[i][j,:,:] = np.eye(self.dhat_in[i]) * np.exp(self.log_theta_sigma2[i])

        return mean_W, Sigma_W

    ## Function to initialize the posterior over log_theta_sigma2
    def init_prior_log_theta_sigma2(self):
        log_theta_sigma2 = np.zeros(self.n_Omega) # utils.get_random(self.n_Omega)

        return log_theta_sigma2

    ## Function to initialize the posterior over hidden layers
    def init_prior_X0(self):
        X0 = np.random.normal(loc = 0.0, scale = 1.0, size = [self.M, self.d_x])

        return X0

    def init_prior_Phi(self):
        N_prior = max(self.dhat_in) # self.init_dataset.Y.shape[0] # max(self.dhat_in)
        Phi = [np.zeros([self.dhat_out[i-1], N_prior]) for i in range(1, self.nl)]

        for t in range(N_prior):
            self.forward_prior()
            self.backward_prior()

            for i in range(1, self.nl): Phi[i-1][:, self.T] = self.backward_layer[i]

            self.update_prior()
            self.T += 1

        Phi.append(self.init_dataset.Y.T)

        return Phi

    def forward_prior(self):

        noise = self.sample_from_noise(prior=True)

        x = self.Xs
        y = self.init_dataset.Y[self.T, :]

        self.forward_layer = [x]
        self.forward_mean = [None]
        self.forward_std = [None]

        ## Forward propagate information from the input to the output through hidden layers
        for i in range(self.nl):
            layer_times_Omega = np.dot(self.forward_layer[i], self.Omega[i])  # x * Omega

            ## Apply the activation function corresponding to the chosen kernel - PHI
            if self.kernel_type == "RBF":
                Phi = 1 / np.sqrt(self.n_rff[i]) * np.concatenate([np.cos(layer_times_Omega), np.sin(layer_times_Omega)], axis=1)
            elif self.kernel_type == "arccosine":
                Phi = 1 / np.sqrt(self.n_rff[i]) * np.maximum(layer_times_Omega, 0.0)

            mean = np.dot(Phi, self.mean_W[i].T)  # mc * dhat_out
            self.forward_mean.append(mean)

            std = np.zeros([self.M, self.dhat_out[i]])
            for j in range(self.dhat_out[i]):
                var = utils.diag(Phi, self.Sigma_W[i][j,:,:]) + self.prior_var
                std[:,j] = np.sqrt(var)
            self.forward_std.append(std)

            F = mean + std * noise[i]
            self.forward_layer.append(F)

        ## Output layer
        mean_out = self.forward_mean[self.nl]
        std_out = self.forward_std[self.nl]

        ll = np.maximum(self.prior_likelihood.log_cond_prob(y, mean_out, std_out), -1e4)
        ell = np.mean(np.average(ll, 0))

        return ell, mean_out

    def backward_prior(self):

        y = self.init_dataset.Y[self.T, :]
        self.backward_layer = [y]

        ## Backward propagate information from the output to the input through hidden layers
        for i in reversed(range(2, self.nl+1)):
            log_weights = self.prior_likelihood.log_cond_prob(self.backward_layer[0], self.forward_mean[i], self.forward_std[i])
            log_weight = np.sum(log_weights, axis=1)
            weight = utils.normalize_weight(log_weight)

            F = np.average(self.forward_layer[i - 1], axis=0, weights = weight)

            if i == 2:
                indices = random.choices(range(self.M), weight, k=self.M) # np.random.choice(range(self.M), size=self.M, p=weight)
                self.Xs = np.array([self.forward_layer[1][m, :self.d_x] for m in indices])

            self.backward_layer.insert(0, F)

        self.backward_layer.insert(0, self.Xhat[-1][0])
        self.Xhat.append(list(self.backward_layer[1:-1]))

    def update_prior(self):
        for i in range(self.nl):
            layer_times_Omega = np.dot(self.backward_layer[i], self.Omega[i])  # n_rff

            ## Apply the activation function corresponding to the chosen kernel - PHI
            if self.kernel_type == "RBF":
                phi = 1 / np.sqrt(self.n_rff[i]) * np.concatenate([np.cos(layer_times_Omega), np.sin(layer_times_Omega)], axis=0)
            elif self.kernel_type == "arccosine":
                phi = 1 / np.sqrt(self.n_rff[i]) * np.maximum(layer_times_Omega,0.0)

            mean = np.dot(phi, self.mean_W[i].T)
            residual = self.backward_layer[i + 1] - mean
            for j in range(self.dhat_out[i]):
                tmp = np.dot(phi, self.Sigma_W[i][j, :, :])
                sufficient = self.prior_var + np.dot(tmp, phi)
                k = tmp / sufficient
                self.mean_W[i][j,:] += k * residual[j]
                self.Sigma_W[i][j,:,:] = np.dot(np.eye(self.dhat_in[i]) - np.dot(k.reshape(-1,1), phi.reshape(1,-1)), self.Sigma_W[i][j,:,:])

    ## Function to initialize the auxiliary parameter r
    def init_prior_r(self):
        r = []
        for i in range(self.nl):
            r_i_ = np.zeros(self.dhat_out[i])
            for j in range(self.dhat_out[i]):
                hidden_layer = self.Phi[i][j, :]
                mean_W = self.mean_W[i][j, :]
                Sigma_W = self.Sigma_W[i][j, :, :]
                Sigma_W_inverse = np.linalg.inv(Sigma_W + 1e-8 * np.eye(self.dhat_in[i]))
                r_i_[j] = max(np.dot(hidden_layer, hidden_layer) - np.dot(np.dot(mean_W, Sigma_W_inverse), mean_W), 1e-2)
            r.append(r_i_)

        return r

    ## Returns Student t's noises for layers
    def sample_from_noise(self, prior=False):
        noise = []
        for i in range(self.nl):
            if prior:
                z = utils.get_normal_samples([self.M, self.dhat_out[i]])
            else:
                nu = self.T - self.dhat_in[i] + 1
                z = utils.get_t_samples([self.M, self.dhat_out[i]], nu)
            noise.append(z)

        return noise

    ## Returns the expected log-likelihood term and prediction
    def forward(self, y):
        noise = self.sample_from_noise()

        x = self.Xs

        ## The representation of the information is based on 2-dimensional ndarrays (one for each layer)
        ## Each slice [i,:] of these ndarrays is one Monte Carlo realization of the value of the hidden units
        ## At layer zero we simply replicate the input matrix X self.mc times
        self.forward_layer = [x]
        self.forward_mean = [None]
        self.forward_std = [None]

        ## Forward propagate information from the input to the output through hidden layers
        for i in range(self.nl):
            nu = self.T - self.dhat_in[i] + 1
            layer_times_Omega = np.dot(self.forward_layer[i], self.Omega[i])  # x * Omega

            ## Apply the activation function corresponding to the chosen kernel - PHI
            if self.kernel_type == "RBF":
                Phi = 1 / np.sqrt(self.n_rff[i]) * np.concatenate([np.cos(layer_times_Omega), np.sin(layer_times_Omega)], axis=1)
            elif self.kernel_type == "arccosine":
                Phi = 1 / np.sqrt(self.n_rff[i]) * np.maximum(layer_times_Omega, 0.0)

            mean = np.dot(Phi, self.mean_W[i].T)  # mc * dhat_out
            self.forward_mean.append(mean)

            a = np.zeros([self.M, self.dhat_out[i]])
            for j in range(self.dhat_out[i]):
                self.Sigma_W[i][j, :, :] += 1e-3 * np.diag(np.diag(self.Sigma_W[i][j, :, :]))
                a[:, j] = self.r[i][j] * (1 + utils.diag(Phi, self.Sigma_W[i][j, :, :]))
            std = np.sqrt(a / nu)
            self.forward_std.append(std)

            F = mean + std * noise[i]
            self.forward_layer.append(F)

        ## Output layer
        mean_out = self.forward_mean[self.nl]
        std_out = self.forward_std[self.nl]

        ## Given the output layer, we compute the conditional likelihood across all samples
        ll = np.maximum(self.reg_likelihood.log_cond_prob(y, mean_out, std_out, nu), -1e4)
        ell = np.mean(np.average(ll,0))

        return ell, mean_out

    def backward(self, y, u = None):
        self.backward_layer = []
        self.backward_layer.insert(0, y)

        ## Backrward propagate information from the output to the input through hidden layers
        for i in reversed(range(2, self.nl+1)):
            nu = self.T - self.dhat_in[i - 1] + 1
            log_weights = self.reg_likelihood.log_cond_prob(self.backward_layer[0], self.forward_mean[i], self.forward_std[i], nu)
            log_weight = np.sum(log_weights, axis=1)
            weight = utils.normalize_weight(log_weight)

            F = np.average(self.forward_layer[i - 1], axis=0, weights = weight)

            if i == 2:
                indices = random.choices(range(self.M), weight, k=self.M)
                self.Xs = np.array([self.forward_layer[1][m, :self.d_x] for m in indices])

            self.backward_layer.insert(0, F)

        self.backward_layer.insert(0, self.Xhat[-1][0])
        self.Xhat.append(list(self.backward_layer[1:-1]))

    def update(self):
        for i in range(self.nl):
            layer_times_Omega = np.dot(self.backward_layer[i], self.Omega[i])  # n_rff

            ## Apply the activation function corresponding to the chosen kernel - PHI
            if self.kernel_type == "RBF":
                phi = 1 / np.sqrt(self.n_rff[i]) * np.concatenate([np.cos(layer_times_Omega), np.sin(layer_times_Omega)], axis=0)
            elif self.kernel_type == "arccosine":
                phi = 1 / np.sqrt(self.n_rff[i]) * np.maximum(layer_times_Omega,0.0)

            mean = np.dot(phi, self.mean_W[i].T)  # dhat_out
            residual = self.backward_layer[i + 1] - mean
            for j in range(self.dhat_out[i]):
                tmp = np.dot(phi, self.Sigma_W[i][j, :, :])
                sufficient = 1 + np.dot(tmp, phi)
                k = tmp / sufficient
                self.r[i][j] += pow(residual[j], 2) / sufficient
                self.mean_W[i][j, :] += k * residual[j]
                self.Sigma_W[i][j, :, :] = np.dot(np.eye(self.dhat_in[i]) - np.dot(k.reshape(-1, 1), phi.reshape(1, -1)), self.Sigma_W[i][j, :, :])

    ## Function that learns the deep GP model sequentially with random Fourier feature approximation
    def learn(self, y, reset=None):
        if reset:

            self.Xs = self.init_prior_X0()
            # self.Xs = utils.get_mvn_samples(mean=[0] * self.d_x, cov=np.diag(np.var(self.Xs, axis=0)),shape=self.M)  # self.Xs - np.average(self.Xs, axis=0)
            self.Xhat = [[np.zeros(self.d_x)]]

            # X0s = np.array([self.Xhat[i][0] for i in range(max(self.dhat_in), len(self.Xhat))])
            # self.Xs = utils.get_mvn_samples(mean=[0] * self.d_x, cov=np.diag(np.var(X0s, axis=0)), shape=self.M)  # self.Xs - np.average(self.Xs, axis=0)
            # self.Xhat = [[np.average(X0s, axis=0)]]

        ## Present one sample to the DGP
        ell, mean_out = self.forward(y)
        self.ell = ell
        self.mean_out = mean_out # np.average(mean_out, axis=0)

        self.backward(y)

        self.update()

        self.T += 1

