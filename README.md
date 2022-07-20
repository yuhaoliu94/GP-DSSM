# README

The variational inference is constructed under Tensorflow 1.14.0. To save space and time, the suggestion is to prefer variational inference as the prior of online learning under high-dimensional cases. To clarify, the "Omega" in the codes is the "v" in the paper while the "W" in the codes is the "theta" in the paper. 

## FLAGS

The code implements particle filterng for Online Ensemble Deep Gaussian Processes (OEDGP) approximated using random Fourier features. The code accepts the following options:


* --nl                  &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &nbsp; Number of layers
* --df                  &emsp; &ensp; &emsp; &emsp; &emsp; &emsp; &emsp; Number of GPs per hidden layer
* --VI                  &emsp; &ensp; &emsp; &emsp; &emsp; &emsp; &emsp; Whether use the variational inference as prior
* --fold                &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; Fold of dataset
* --seed                &emsp; &emsp; &emsp; &emsp; &emsp; &ensp; Seed for Tensorflow and Numpy
* --n_rff               &emsp; &emsp; &emsp; &emsp; &emsp; &ensp; Number of random features
* --dataset             &emsp; &emsp; &emsp; &emsp; &ensp; Name of dataset
* --MC_test             &emsp; &ensp; &emsp; &emsp; &emsp; Number of Monte Carlo samples for online prediction
* --mc_test             &emsp; &emsp; &emsp; &emsp; &nbsp; Number of Monte Carlo samples for VI predictions and Number of candidate models
* --MC_train            &emsp; &ensp; &emsp; &emsp; &emsp; Number of Monte Carlo samples for online training
* --mc_train            &emsp; &emsp; &emsp; &emsp; &nbsp; Number of Monte Carlo samples for VI training
* --duration            &emsp; &emsp; &emsp; &emsp; &nbsp; Duration in minutes
* --optimizer           &emsp; &emsp; &emsp; &emsp; Optimizer: adam, adagrad, adadelta, or sgd
* --batch_size          &emsp; &emsp; &emsp; &nbsp; Batch size for VI
* --kernel_type         &emsp; &emsp; &emsp; Kernel: RBF, arccosine, or identity
* --less_prints         &emsp; &emsp; &emsp; &nbsp; Disables evaluations during the training steps
* --theta_fixed         &emsp; &emsp; &emsp; &nbsp; Number of iterations to keep theta fixed at the beginning
* --N_iterations        &emsp; &emsp; &emsp; Number of iterations (samples) to train the online DGP model
* --n_iterations        &emsp; &emsp; &emsp; Number of iterations (batches) to train the VI DGP model
* --display_step        &emsp; &emsp; &emsp; Display progress every display_step iterations
* --learning_rate       &emsp; &emsp; &nbsp; &nbsp; Learning rate for optimizers
* --local_reparam       &emsp; &emsp; &nbsp; Use the local reparameterization trick
* --q_Omega_fixed       &emsp; &ensp; Number of iterations to keep posterior of Omega fixed at the beginning


## EXAMPLES

Here are two examples to run the OEDGP model on regression and classification tasks:

### REGRESSION
```
python experiments/oedgp_rff_regression.py --seed=12345 --dataset=powerplant --fold=1 --q_Omega_fixed=1000 \
--theta_fixed=4000 --optimizer=adam --nl=2 --df=3 --learning_rate=0.001 --n_rff=50 \
--batch_size=200 --mc_train=100 --mc_test=100 --n_iterations=100000 --display_step=250 --duration=60 \
--kernel_type=RBF --VI=True --MC_train=100 --MC_test=100 --local_reparam=True
```

### CLASSIFICATION
```
python experiments/oedgp_rff_classification.py --seed=12345 --dataset=credit --fold=1 --q_Omega_fixed=1000 \
--theta_fixed=4000 --optimizer=adam --nl=2 --df=3 --learning_rate=0.001 --n_rff=50 \
--batch_size=200 --mc_train=100 --mc_test=100 --n_iterations=100000 --display_step=250 --duration=60 \
--kernel_type=arccosine --VI=False --MC_train=100 --MC_test=100 --local_reparam=True
```
