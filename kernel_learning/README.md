## Code Overview

tf_kernel_network.py: Tensorflow compatible code to create and optimize Nystrom and Random Fourier feature maps based on the Discriminant Information (DI) objective.

tf_kernel.py: Tensorflow compatible code to produce Gram (kernel) matrices from tensors.

tf_classifier.py: Code to train linear and kernel based classifiers with stochastic gradient methods implemented by tensorflow. 

nystrom: Utilizes numpy and scipy libraries to produce multiple variants of Nystrom approximations proposed in the literature.

random_fourier: Utilizes numpy and scipy libraries to produce Random Fourier approximations.

linear_ridge: Utilizes numpy and scipy libraries to train Ridge Regression predictors.

kernel_ridge: Utilizes numpy and scipy libraries to train Kernel Ridge Regression predictors.

You may find example code to optimize Nystrom and Random Fourier feature maps in the "examples" folder. Note that the example codes use fixed learning rate rules, though adaptive learning rates were used in our experiments. Adaptive learning rates can be implemented by keeping track of the objective values, which the example code already does.
