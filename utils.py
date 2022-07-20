import numpy as np
from scipy.stats import t, norm
import tensorflow as tf

## Draw prior hyper-parameters
def get_random(size):
    # stack = [1] #pow(10, np.linspace(-4,6,11))
    # return np.random.choice(stack, size)
    # return np.random.uniform(1e-4,10,size)
    return np.exp(np.random.uniform(-4, 4, size))

## Log-density of a univariate Gaussian distribution
def log_norm_pdf(x, loc=0.0, scale=0.0):
    return norm.logpdf(x, loc=loc, scale=scale)

## Log-density of a univariate t distribution
def log_t_pdf(x, loc=0.0, scale=0.0, df=3):
    return t.logpdf(x, df=df, loc=loc, scale=scale)

## Draw an array of multivariate normal
def get_mvn_samples(mean, cov, shape):
    return np.random.multivariate_normal(mean=mean, cov=cov, size=shape)

## Draw an array of standard normal
def get_normal_samples(shape):
    return np.minimum(np.random.normal(size=shape), 2)

## Draw an array of standard student's t
def get_t_samples(shape, nu):
    return np.minimum(np.random.standard_t(df=nu, size=shape), 2)

## Calculate the phi^\top \Sigma \phi
def diag(Phi, Sigma):
    if Phi.ndim == 1:
        tmp = np.dot(Phi, Sigma)
        return np.dot(tmp, Phi)
    else:
        tmp = np.dot(Phi, Sigma)
        var = [np.dot(tmp[mc,:], Phi[mc,:]) for mc in range(Phi.shape[0])]
        return np.array(var)

## Normalize log weights
def normalize_weight(log_weight):
    log_weight -= max(log_weight)
    weight = np.exp(log_weight)
    weight /= sum(weight)
    return weight

## Log-sum operation
def logsumexp(vals, dim=None):
    m = np.max(vals, dim)
    if dim is None:
        return m + np.log(np.sum(np.exp(vals - m), dim))
    else:
        return m + np.log(np.sum(np.exp(vals - np.expand_dims(m, dim)), dim))


## Get flags from the command line
def get_flags():
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_integer('display_step', 100, 'Display progress every FLAGS.display_step iterations')
    flags.DEFINE_integer('mc', 1000, 'Number of Monte Carlo samples for predictions')
    flags.DEFINE_integer('n_candidates', 100, 'Number of Monte Carlo samples for models')
    flags.DEFINE_integer('n_rff', 50, 'Number of random features for each layer')
    flags.DEFINE_string('df', "1", 'Number of GPs per hidden layer')
    flags.DEFINE_integer('nl', 2, 'Number of layers')
    flags.DEFINE_string('kernel_type', "RBF", 'arccosine')
    flags.DEFINE_integer('duration', 100000, 'Duration of job in minutes')
    flags.DEFINE_float('prior_var', 0.001, 'variance on prior inference')

    # Flags for use in cluster experiments
    tf.app.flags.DEFINE_string("dataset", "", "Dataset name")
    tf.app.flags.DEFINE_string("fold", "1", "Dataset fold")
    tf.app.flags.DEFINE_integer("seed", 12345, "Seed for random tf and np operations")
    tf.app.flags.DEFINE_boolean("less_prints", False, "Disables evaluations involving the complete dataset without batching")
    tf.app.flags.DEFINE_integer("N_iterations", 1, "Number of iterations")
    
    return FLAGS
