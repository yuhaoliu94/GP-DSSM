# README

Gaussian Process-based Deep State-Space Model. The current version only supports the continuous observation processes.

## FLAGS

The code implements particle filterng for Gaussian Process-based State-Space Models (GP-DSSMs) approximated using random Fourier features. The code accepts the following options:


* --nl                  &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &nbsp; Number of layers, including the observatioin layer
* --df                  &emsp; &ensp; &emsp; &emsp; &emsp; &emsp; &emsp; Number of GPs for hidden layers
* --mc                  &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &nbsp; Number of Monte Carlo samples for predictions
* --fold                &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; Fold of dataset
* --seed                &emsp; &emsp; &emsp; &emsp; &emsp; &ensp; Seed for Tensorflow and Numpy
* --n_rff               &emsp; &emsp; &emsp; &emsp; &emsp; &ensp; Number of random features
* --dataset             &emsp; &emsp; &emsp; &emsp; &ensp; Name of dataset
* --duration            &emsp; &emsp; &emsp; &emsp; &nbsp; Duration in minutes
* --prior_var           &emsp; &emsp; &emsp; &emsp; Variance on prior inference
* --kernel_type         &emsp; &emsp; &emsp; Kernel: RBF or arccosine
* --less_prints         &emsp; &emsp; &emsp; &nbsp; Disables evaluations during the training steps
* --n_candidates        &emsp; &emsp; &emsp; Number of Monte Carlo samples for models
* --display_step        &emsp; &emsp; &emsp; Display progress every display_step iterations

## EXAMPLES

Here is one examples to run the GPSSM model on regression tasks:

### REGRESSION
```
python3 experiments/edgpssm.py --seed=12345 \
--mc=1000 --duration=1200 --display_step=10 --n_rff=50 --less_prints=False \
--dataset=synthetic --kernel_type=RBF --n_candidates=100 --prior_var=0.001 --nl=3 --df="1,3" --fold=1
