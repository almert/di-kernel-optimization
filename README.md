# Discriminant Information Based Kernel Optimization
### Python codes for optimizing kernel feature maps via the Discriminant Information criterion

## Overview
This project contains the Python codes for optimizing Nystrom and Random Fourier feature maps based on the Discriminant Information (DI) criterion. The codes were used to perform the experiments reported in [[1](#citation)] (https://arxiv.org/abs/1909.10432).

## Usage
The code can be used to apply Random Fourier and Nystrom kernel feature maps on the data, as well as to optimize such feature maps for particular supervised learning tasks. 

The main functionality can be found within tf_kernel.py and tf_kernel_network.py files. The other files can be used to create kernel feature maps and kernel based predictors. Files, whose names start with "tf" are useful for producing feature maps and training objectives compatible with tensorflow.

Please cite [[1](#citation)] in your work when using these codes in your experiments.

## Dependencies
The .py files that don't start with "tf" should be compatible with all versions of scikit-learn and numpy that come with Python 3.5 or 3.6. The tensorflow parts of the code were tested on tensorflow 1.12.0. To make them compatible with the most recent versions of tensorflow, replace "import tensorflow as tf" with "import tensorflow.compat.v1 as tf" and add the command "tf.compat.v1.disable_eager_execution()" to your code. 

## License
Princeton University

## Citation
```
[1] Mert Al, Zejiang Hou, and Sun-Yuan Kung. Scalable Kernel Learning via the Discriminant Information. IEEE International Conference on Acoustics, Speech and Signal Processing. IEEE, 2020.
```

BibTeX format:
```
@article{al2020scalable,
  title     = {Scalable Kernel Learning via the Discriminant Information},
  author    = {Mert Al and
               Zejiang Hou and
               Sun-Yuan Kung},
  booktitle = {Proc.~IEEE International Conference on Acoustics, Speech, and Signal Processing},
  year      = {2020},
  month     = {May},
  organization = {IEEE}
  url       = {https://arxiv.org/abs/1909.10432},
}
```
