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



## Installation

**Required:** Pybind11 package.

Run the following line to install pybind11.

Anaconda:
```
conda install -c conda-forge pybind11
```

Pip:
```
pip install pybind11
```

## Data generation
### required parameters 
Potential function (-p), number of samples (-ns), mesh grid (-nx), time step size (-dt), num of steps (-nt), GRF parameters (-alp, -tau).

for example
```
python data_generation.py -p gaussian -ns 2 -nx 64 -nt 10 -dt 0.0005 -alp 4 -tau 10
```

