import os
import sys

sys.path.append(".")
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


from datasets import DataSet
import utils
from edgpssm_rff import edgpssm
import numpy as np

def import_dataset(dataset, fold):
    Y = np.loadtxt('FOLDS/' + dataset + '_' + fold, delimiter=' ')
    if Y.ndim == 1: Y = np.reshape(Y, (-1, 1))

    data = DataSet(Y, dataset, fold)

    return data

if __name__ == '__main__':
    FLAGS = utils.get_flags()

    ## Set random seed for numpy operations
    np.random.seed(FLAGS.seed)
    dataset = import_dataset(FLAGS.dataset, FLAGS.fold)

    df = FLAGS.df.split(",")
    df = [int(char) for char in df]

    ## Main dgp object
    edgp_ssm = edgpssm(dataset, dataset.Dout, FLAGS.nl, FLAGS.n_rff, df, FLAGS.mc, FLAGS.kernel_type, FLAGS.n_candidates, FLAGS.prior_var)

    ## Learning
    edgp_ssm.learn(FLAGS.display_step, FLAGS.less_prints, FLAGS.N_iterations)


