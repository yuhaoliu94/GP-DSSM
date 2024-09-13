# README

This repository contains code for the Gaussian Process-based Deep State-Space Model (GP-DSSM) proposed in the paper:

Yuhao Liu, Marzieh Ajirak, and Petar M. Djurić. "Sequential estimation of Gaussian process-based deep state-space models." IEEE Transactions on Signal Processing (2023).

The work is [https://ieeexplore.ieee.org/abstract/document/10216326](https://ieeexplore.ieee.org/abstract/document/10216326) or in arXiv [https://arxiv.org/pdf/2301.12528](https://arxiv.org/pdf/2301.12528).

09/13/2024 - Added several toy examples.

The case 1 and 2 shows a toy example with one-hidden-layer, where the diagram is as follows:
![](https://github.com/yuhaoliu94/GP-DSSM/blob/main/notebook/README/case1_2.png?raw=true)

The case 3 shows a one-hidden-layer example with control input, where the diagram is as follows:
![](https://github.com/yuhaoliu94/GP-DSSM/blob/main/notebook/README/case3.png?raw=true)

The case 4 shows a one-hidden-layer example with feed-forward control input, where the diagram is as follows:
![](https://github.com/yuhaoliu94/GP-DSSM/blob/main/notebook/README/case4.png?raw=true)

There are other cases not implemented in the notebook but are easy to construct. Users can customize their models in `models.py`.

1. We can construct a multiple-hidden-layer SSM where the hidden-layer also has transitions:
![](https://github.com/yuhaoliu94/GP-DSSM/blob/main/notebook/README/case5.png?raw=true)

2. We can construct a multiple-hidden-layer SSM where the hidden-layer does not have transitions:
![](https://github.com/yuhaoliu94/GP-DSSM/blob/main/notebook/README/case6.png?raw=true)

3. We can construct a multiple-hidden-layer SSM without feed-forward input:
![](https://github.com/yuhaoliu94/GP-DSSM/blob/main/notebook/README/case7.png?raw=true)

4. We can construct a multiple-hidden-layer SSM without input:
![](https://github.com/yuhaoliu94/GP-DSSM/blob/main/notebook/README/case8.png?raw=true)

06/28/2024 - Currently, it is structured so that DGPs are learned using gradient optimization, and a loss function of interest is displayed on the test error every fixed number of iterations.