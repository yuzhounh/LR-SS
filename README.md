# Logistic Regression with Sparse and Smooth Regularizations

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This repository contains the implementation of our paper "Incorporating Symmetric Smooth Regularizations into Sparse Logistic Regression for Classification and Feature Extraction".

## Overview

This work focuses on solving the optimization problem for logistic regression with sparse and smooth regularizations (LR-SS):

$$\max_{\mathbf{w}}\ln P(\mathbf{y}|\mathbf{X},\mathbf{w})-\lambda_1\|\mathbf{w}\|_1-\frac{\lambda_2}{2}\mathbf{w}^T\mathbf{Q}\mathbf{w}$$

where $\mathbf{w}^T \mathbf{Q} \mathbf{w}$ is the smooth regularization, $\lambda_1$ and $\lambda_2$ are non-negative regularization parameters controlling the strength of the Laplacian and smooth priors, respectively.

## Special Cases

By adjusting the parameters, LR-SS can degenerate into the following algorithms:

1. When $\lambda_1 = 0$ and $\lambda_2 = 0$, LR-SS degenerates into standard logistic regression, denoted as LR.

2. When $\lambda_1 = 0$, $\lambda_2 \neq 0$ and $\mathbf{Q} = \mathbf{I}$, LR-SS degenerates into logistic regression with L2-norm regularization, denoted as LR-L2.

3. When $\lambda_1 \neq 0$ and $\lg(\lambda_2) = 0$, LR-SS degenerates into logistic regression with L1-norm regularization (standard sparse logistic regression), denoted as LR-L1.

4. When $\lambda_1 \neq 0$, $\lambda_2 \neq 0$ and $\mathbf{Q} = \mathbf{I}$, LR-SS degenerates into logistic regression with ElasticNet regularization, denoted as LR-ElasticNet.

5. When $\lambda_1 \neq 0$, $\lambda_2 \neq 0$, $\mathbf{Q} = \mathbf{Q}^{(1)}$ and $\varepsilon = 1$, LR-SS degenerates into logistic regression with GraphNet regularization, denoted as LR-GraphNet.

6. When $\lambda_1 \neq 0$, $\lambda_2 \neq 0$ and $\mathbf{Q} = \mathbf{Q}^{(1)}$, the first form of LR-SS is obtained, denoted as LR-SS1.

7. When $\lambda_1 \neq 0$, $\lambda_2 \neq 0$ and $\mathbf{Q} = \mathbf{Q}^{(2)}$, the second form of LR-SS is obtained, denoted as LR-SS2.

## Usage

1. Download the four real-world datasets:
   - [DistalPhalanxOutlineCorrect](https://www.timeseriesclassification.com/description.php?Dataset=DistalPhalanxOutlineCorrect)
   - [GunPoint](https://timeseriesclassification.com/description.php?Dataset=GunPoint)  
   - [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist)
   - [MNIST](https://yann.lecun.com/exdb/mnist/) or [MNIST Alternative](https://github.com/cvdfoundation/mnist)
2. Place and extract the downloaded datasets in the `data` folder
```
data/
├── DistalPhalanxOutlineCorrect/
│   ├── DistalPhalanxOutlineCorrect.txt
│   ├── DistalPhalanxOutlineCorrect_TEST.arff
│   ├── DistalPhalanxOutlineCorrect_TEST.ts
│   ├── DistalPhalanxOutlineCorrect_TEST.txt
│   ├── DistalPhalanxOutlineCorrect_TRAIN.arff
│   ├── DistalPhalanxOutlineCorrect_TRAIN.ts
│   └── DistalPhalanxOutlineCorrect_TRAIN.txt
├── FashionMNIST/
│   ├── t10k-images-idx3-ubyte.gz
│   ├── t10k-labels-idx1-ubyte.gz
│   ├── train-images-idx3-ubyte.gz
│   └── train-labels-idx1-ubyte.gz
├── GunPoint/
│   ├── GunPoint.txt
│   ├── GunPoint_TEST.arff
│   ├── GunPoint_TEST.ts
│   ├── GunPoint_TEST.txt
│   ├── GunPoint_TRAIN.arff
│   ├── GunPoint_TRAIN.ts
│   └── GunPoint_TRAIN.txt
└── MNIST/
    ├── t10k-images.idx3-ubyte.gz
    ├── t10k-labels.idx1-ubyte.gz
    ├── train-images.idx3-ubyte.gz
    └── train-labels.idx1-ubyte.gz
```
3. Download the code from this repository
4. Run `main.m` in MATLAB to:
   - Compare smoothing matrices
   - Generate and analyze simulated datasets
   - Load and process real-world datasets
   - Perform grid search for optimal parameters on simulated data
   - Visualize weight vectors learned by different algorithms
   - Perform Bayesian optimization on both simulated and real datasets
  
This repository has been tested on MATLAB R2024a.

Below are the weight vectors learned by different algorithms on the simulated dataset.

![Weight vectors learned by different algorithms](weight_vectors.svg)

  
## Contact

For questions and feedback, please contact: **Jing Wang** (wangjing@xynu.edu.cn). 
