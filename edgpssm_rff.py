from __future__ import print_function

import os
os.environ["PYTHONUNBUFFERED"] = "1"
import numpy as np
import matplotlib

matplotlib.use('Agg')
import likelihoods
import time
import losses
from datasets import DataSet
from dgpssm_rff import dgpssm
import pickle
import matplotlib.pyplot as plt

current_milli_time = lambda: int(round(time.time() * 1000))


class edgpssm(object):

    def __init__(self, dataset, d_out, n_layers, n_rff, df, M, kernel_type, n_candidates, prior_var):
        """
        :param d_in: Dimensionality of the input
        :param d_out: Dimensionality of the output
        :param n_layers: Number of hidden layers
        :param n_rff: Number of random features for each layer
        :param df: Number of GPs for each layer
        :param kernel_type: Kernel type: currently only random Fourier features for RBF and arccosine kernels are implemented
        :param n_candidates: The number of candidate models to ensemble
        :param break_dim: Which part is regression or classification
        """
        self.reg_likelihood = likelihoods.Student_t()
        self.class_likelihood = likelihoods.Softmax()
        self.kernel_type = kernel_type
        self.dataset = dataset

        ## These are all scalars
        self.nl = n_layers  ## Number of hidden layers
        self.nc = n_candidates  ## Number of candidate models
        self.M = M

        ## These are arrays to allow flexibility in the future
        self.n_rff = n_rff * np.ones(n_layers, dtype=np.int32)
        self.df = df * np.ones(n_layers - 1, dtype=np.int32)
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

        ## Initialize lists of ensemble models
        init_dataset = DataSet(dataset.Y[:max(self.dhat_in), :], dataset.name, dataset.fold)
        self.ensemble = [dgpssm(init_dataset, d_out, n_layers, n_rff, df, M, kernel_type, prior_var, i) for i in range(n_candidates)]
        self.weight = np.array([1 / n_candidates for _ in range(n_candidates)])
        self.pred = [None for _ in range(n_candidates)]


    def learn(self, display_step=100, less_prints=False, N_iterations = 1):
        print(">>> Ensemble Online learning starts.")

        mnll_train = loss = 0
        total_train_time = 0

        Y = self.dataset.Y
        num_samples = Y.shape[0]

        reg_loss = losses.RootMeanSqError(self.dataset.Dout)

        real = []
        prediction = []
        mnll = []
        time = []
        rmse = []

        ## Present data to DGP n_iterations times
        for t in range(max(self.dhat_in), num_samples * N_iterations):
            start_train_time = current_milli_time()

            ## Present candidate models to the DGP
            ells = []
            outs =[]

            reset = True if t % num_samples == 0 else False
            if reset: mnll_train = loss = 0
            for i in range(self.nc):
                if self.weight[i] > 1e-16:
                    self.ensemble[i].learn(Y[t % num_samples], reset)
                    ells.append(self.ensemble[i].ell)
                    outs.append(self.ensemble[i].mean_out)
                else:
                    self.weight[i] = 0
                    ells.append(0)
                    outs.append(np.zeros([self.M, self.dhat_out[-1]]))

            ell_mean = np.average(ells, weights=self.weight)
            out_mean = np.average(outs, weights=self.weight, axis=0)

            t0 = 2000
            if t < t0:
                t_train = t - max(self.dhat_in) if t // num_samples == 0 else t % num_samples
                mnll_train = (mnll_train * t_train + ell_mean) / (t_train + 1)
                loss = np.sqrt((pow(loss, 2) * t_train + pow(reg_loss.eval(Y[t % num_samples], out_mean), 2)) / (t_train + 1))

            if t == t0:
                mnll_train = loss = 0

            if t >= t0:
                t_train = t - t0 if t // num_samples == 0 else t % num_samples
                mnll_train = (mnll_train * t_train + ell_mean) / (t_train + 1)
                loss = np.sqrt((pow(loss, 2) * t_train + pow(reg_loss.eval(Y[t % num_samples], out_mean), 2)) / (t_train + 1))

                for i in range(self.nc):
                    self.weight[i] *= np.exp(self.ensemble[i].ell)
                self.weight /= sum(self.weight)

            total_train_time += current_milli_time() - start_train_time

            if t // num_samples == N_iterations - 1:
                prediction.append(np.average(out_mean, 0))
                real.append(Y[t % num_samples])
                mnll.append(-mnll_train)
                time.append(total_train_time)
                rmse.append(loss)

            ## Display logs every "FLAGS.display_step" iterations
            if t % display_step == 0 or t == num_samples * N_iterations - 1:
                if less_prints: print(">>> t=" + repr(t), end = " ")
                else:
                    print(">>> t=" + repr(t) + " n=" + repr(t // num_samples) + " mnll_train=" + repr(-mnll_train) + " loss=" + repr(loss) + " time=" + repr(total_train_time), end="  ")
                print("")

            # Stop after a given budget of minutes is reached
            # if (total_train_time > 1000 * 60 * duration):
            #     break

        print(">>> Ensemble Online learning ends.")

        path = "Results/" + self.dataset.name  + "_" + str(self.dataset.fold)
        if not os.path.exists(path):
            os.mkdir(path)
            os.mkdir(path + "/Model")
            os.mkdir(path + "/X")
            os.mkdir(path + "/Y")

        real = np.array(real)
        prediction = np.array(prediction)
        mnll = np.array(mnll)
        rmse = np.array(rmse)
        time = np.array(time)

        for i in range(Y.shape[1]):
            plt.plot(real[:,i], label="real")
            plt.plot(prediction[:,i], label="pred")
            plt.legend()
            plt.savefig(path + '/Y/test' + "_" + str(i) + '.png')
            plt.close()

        for filename in os.listdir(path + "/X"):
            os.remove(path + "/X/" + filename)

        i = np.argmax(self.weight)
        for l in range(self.nl-1):
            Xhat_l = np.array([self.ensemble[i].Xhat[t][l] for t in range(1, num_samples+1)])
            np.savetxt(path + "/X/Xhat" + "_FOLD_" + str(self.dataset.fold) + "_MODEL_" + str(i) + "_LAYER_" + str(l), Xhat_l)

        for filename in os.listdir(path + "/Model"):
            os.remove(path + "/Model/" + filename)

        i = np.argmax(self.weight)
        with open(path + "/Model/Class" + "_FOLD_" + str(self.dataset.fold) + "_MODEL_" + str(i) + ".pkl", 'wb') as p:
            pickle.dump(self.ensemble[i], p)

        np.savetxt(path + "/Y/Prediction" + "_FOLD_" + str(self.dataset.fold), prediction)
        np.savetxt(path + "/Y/MNLL" + "_FOLD_" + str(self.dataset.fold), mnll)
        np.savetxt(path + "/Y/LOSS" + "_FOLD_" + str(self.dataset.fold), rmse)
        np.savetxt(path + "/Y/Time" + "_FOLD_" + str(self.dataset.fold), time)
        print(">>> Model Saved")
