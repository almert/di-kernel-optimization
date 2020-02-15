## Code Overview

tf_kernel_network.py: Tensorflow compatible code to create and optimize Nystrom and Random Fourier feature maps based on the Discriminant Information (DI) objective.

tf_kernel.py: Tensorflow compatible code to produce Gram (kernel) matrices from tensors.

tf_classifier.py: Code to train linear and kernel based classifiers with stochastic gradient methods implemented by tensorflow. 

nystrom: Utilizes numpy and scipy libraries to produce multiple variants of Nystrom approximations proposed in the literature.

random_fourier: Utilizes numpy and scipy libraries to produce Random Fourier approximations.

linear_ridge: Utilizes numpy and scipy libraries to train Ridge Regression predictors.

kernel_ridge: Utilizes numpy and scipy libraries to train Kernel Ridge Regression predictors.
