# DI Kernel Optimization
### Python codes for optimizing kernel mappings based on the Discriminant Information criterion

## Overview
This project contains the Python codes for optimizing Nystrom and Random Fourier feature maps based on the Discriminant Information (DI) criterion. The codes were used to perform the experiments reported in [[1](#citation)] (https://arxiv.org/abs/1909.10432).

## Usage
The code can be used to apply Random Fourier and Nystrom kernel features on data. Files, whose names start with "tf" are useful for producing feature maps and training objectives compatible with tensorflow.

Please cite [[1](#citation)] in your work when using these codes in your experiments.

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
  volume    = {abs/1909.10432},
  year      = {2020},
  url       = {https://arxiv.org/abs/1909.10432},
}
```
