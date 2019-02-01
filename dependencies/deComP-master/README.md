# deComP
[![Travis Status for fujii-team/deComP](https://travis-ci.org/fujii-team/deComP.svg?branch=master)](https://travis-ci.org/fujii-team/deComP)


Python Library for Large Scale Matrix Decomposition with GPU.

## What is deComP

deComP is a compilation of matrix decomposition and deconvolution algorithms,
especially for large scale data.

We compiled (and updated) several algorithms that utilizes numpy's
parallelization capacity as much as possible.

Furthermore, deComP is also compatible to
[`CuPy`](https://github.com/cupy/cupy),
which gives `numpy`-like interface for gpu computing.


## Matrix decomposition

Matrix decomposition problem is the following optimization problem,
<img src="http://latex.codecogs.com/gif.latex?
x=\text{argmin}_{W, H} \left[
\frac{1}{2}\sum_{i, j} \left|y_{ij} - \sum_k c_{ik} \phi_{kj}\right|^2+\alpha \sum_k \left|c_{ik}\right|\right]
" border="0"/>,  
where Y is a given data matrix, with the shape of [n_samples, n_features].
C is the excitation matrix [n_samples, n_latent]
while
<img src="http://latex.codecogs.com/gif.latex?\Phi" border="0"/>
is a basis matrix [n_latent, n_features].

Therefore, the matrix decomposition is a model to find
the best matched basis <img src="http://latex.codecogs.com/gif.latex?\Phi" border="0"/> for data Y.

Sometimes, some regularization is applied to the excitation matrix
(the second term of the above equation), so that C becomes sparse and <img src="http://latex.codecogs.com/gif.latex?\Phi" border="0"/> can be over-complete.


## Implemented models

Currently, we implemented

- [Dictionary Learning](decomp/dictionary_learning.py)  
- [Non-negative Matrix Factorization](decomp/nmf.py)

All the models support complex values as well as real values.
It also supports missing values.

## Algorithms

Most of the above algorithms uses iterations of conditional optimization,
i.e.
first optimize C with fixed <img src="http://latex.codecogs.com/gif.latex?\Phi" border="0"/> and second optimize <img src="http://latex.codecogs.com/gif.latex?\Phi" border="0"/> with fixed C.

The optimization of C matrix is essentially LASSO problem.
We also implemented GPU version of the Lasso solvers.


## Requirements

The official requirements are only `numpy` and `chainer`.
However, in order to work on GPU, we recommend to install `cupy`.
