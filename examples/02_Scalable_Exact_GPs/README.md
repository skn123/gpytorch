# Exact GPs with Scalable Inference

In GPyTorch, Exact GP inference is still our preferred approach to large regression datasets.
By coupling GPU acceleration with [BlackBox Matrix-Matrix Inference](./Simple_GP_Regression_With_LOVE_Fast_Variances_CUDA.ipynb)
and [LancZos Variance Estimates](./Simple_GP_Regression_With_LOVE_Fast_Variances_CUDA.ipynb),
GPyTorch can perform inference on datasets with over 1,000,000 data points while making very few approximations.

## How GPyTorch Scales Exact GPs
- [GP Regression (CUDA) with Fast Variances (LOVE)](./Simple_GP_Regression_With_LOVE_Fast_Variances_CUDA.ipynb)
  - This notebook demonstrates [LOVE](https://arxiv.org/pdf/1803.06058.pdf), a technique to rapidly speed up predictive variance computations.
    Check out this notebook to see how to use LOVE in GPyTorch, and how it compares to standard variance computations.

## Exact GPs with GPU Acceleration

## Using Kernel Approximations

## Exploiting Kernel Structure 
