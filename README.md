# bfm-fast-diffusion

## Description

Codes for solving the following problem:
$$
\rho^{n+1} = \text{arg}\min_{\rho} \int \rho \log \rho dx + V(\rho) + \frac{1}{2\tau} W^2_2(\rho, \rho^n)
$$
using the back-and-forth method (BFM).

Details of the algorithm can be found in this paper: [The back-and-forth method for Wasserstein gradient flows,
Matt Jacobs, Wonjun Lee, and Flavien Léger](https://arxiv.org/pdf/2011.08151.pdf).

The official code website: https://wasserstein-gradient-flows.netlify.app



## You need to install PyBind package
```
conda install -c conda-forge pybind11
```
