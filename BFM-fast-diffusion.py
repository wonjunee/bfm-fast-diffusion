# %%
# Run if you are on Google Colab to install the Python bindings
import os
os.system('bash compile.sh')

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dctn, idctn
import bfmgf

# %% [markdown]
# # Poisson solver with Neumann boundary condition

# Initialize Fourier kernel
def initialize_kernel(n1, n2, dy):
    xx, yy = np.meshgrid(np.linspace(0,np.pi,n1,False), np.linspace(0,np.pi,n2,False))
    # kernel = 2*n1*n1*(1-np.cos(xx)) + 2*n2*n2*(1-np.cos(yy))
    kernel = 2*(1-np.cos(xx))/(dy*dy) + 2*(1-np.cos(yy))/(dy*dy)
    kernel[0,0] = 1     # to avoid dividing by zero
    return kernel

# 2d DCT
def dct2(a):
    return dctn(a, norm='ortho')

# 2d IDCT
def idct2(a):
    return idctn(a, norm='ortho')

# Solving Poisson
#   - Δ u = f
#   output: u = (-Δ)⁻¹ f
def solve_poisson(u, f, kernel,theta1,theta2):
    n = u.shape[0]
    u[:] = 0
    workspace = np.copy(f)
    workspace[0,0] = 1
    workspace = dct2(workspace) / (theta1 + theta2 * kernel)
    workspace[0,0] = 0
    u += idct2(workspace)

# %%
n=256
xx, yy = np.meshgrid(np.linspace(0.5/n,1-0.5/n,n),np.linspace(0.5/n,1-0.5/n,n))
# Testing poisson solver
kernel = initialize_kernel(n, n, 1.0/n)
u = np.zeros((n,n))
f = np.cos(xx * np.pi) * np.cos(yy * np.pi)
solve_poisson(u,f,kernel,0,1)
plt.imshow(u)
plt.title(f"error: {np.mean( (u - u.mean() - (f - f.mean()) * (0.5/(np.pi)**2))**2 )}")

# %% [markdown]
# # Derivation of the formulation
# $$
# \begin{align}
#     &\min_{\rho} \tau U(\rho) + \frac{1}{2} W^2_2(\rho, \rho^n)\\
#     =&\min_{\rho} \max_{\phi} \tau U(\rho) + \int \phi^c d\rho^n + \int \phi d\rho\\
#     =&\max_{\phi} \int \phi^c d\rho^n - \left(\max_{\rho} \int (-\phi) d\rho - \tau U(\rho) \right)\\
#     =&\max_{\phi} \int \phi^c d\rho^n - \tau U^*(-\phi/\tau)
# \end{align}
# $$
# where
# $$
#     \phi^c(x) = \min_{y\in\Omega} \frac{\|x-y\|^2}{2} - \phi(y).
# $$

# %% [markdown]
# # Putting all together
# 
# We have all the ingredients to compute
# $$
#     J(\phi) = \int \phi^c d\mu - U^*(-\phi)
# $$
# 
# $$
#     I(\psi) = \int \psi d\mu - U^*(-\psi^c)
# $$
# 
# 
# ### FLT2D
# 
# Computing $\phi^c$ from $\phi$. The formulation in the code is
# $$
#     \phi^c(x) = \min_{y\in\Omega} \frac{\|x-y\|^2}{2} - \phi(y).
# $$
# 
# ```python
# flt2d = bfmgf.FLT2D(n,n)
# tau = 0.1
# flt2d.find_c_concave(psi,phi,tau)
# ```
# 
# ### Pushforward
# 
# It computes either $ T_\# \mu \quad \text{or} \quad S_\# \delta U^*(\phi).$
# ```python
# method = bfmgf.BFM(n,n)
# push = np.zeros((n,n))
# method.compute_push_forth(push, psi, mu)
# ```
# 
# ### Poisson Solver
# 
# It computes $u$ from 
# $$- \Delta u = \delta U^*(- \phi) - T_\#\mu = \mu - S_\# \delta U^*(-\phi).$$
# ```python
# kernel = initialize_kernel(n, n, 1.0/n)
# u = np.zeros((n,n))
# f = np.cos(xx * np.pi) * np.cos(yy * np.pi)
# solve_poisson(u,f,kernel)
# ```
# 
# ### Computing DU star
# ```python
# bfmgf.calculate_DUstar(DUstar, V, phi, n, n, tau)
# ```
# 
# 

# %%
def iterate_forward(bfmgf, flt2d, method, push, psi, phi, mu, DUstar, V, kernel, n, tau, theta1, theta2):
    flt2d.find_c_concave(psi, phi, tau)
    flt2d.find_c_concave(phi, psi, tau)

    # bfmgf.calculate_DUstar(DUstar, V, phi, n, n, tau)
    DUstar = np.exp(-phi-V)
    DUstar /= DUstar.mean()
    
    method.compute_push_forth(push, phi, psi, mu)

    u = np.zeros((n,n))
    f = - push + DUstar
    solve_poisson(u,f,kernel,theta1,theta2)

    phi[:] = phi + u
    

def iterate_backward(bfmgf, flt2d, method, push, psi, phi, mu, DUstar, V, kernel, n, tau, theta1, theta2):
    flt2d.find_c_concave(psi, phi, tau)
    
    # bfmgf.calculate_DUstar(DUstar, V, phi, n, n, tau)
    DUstar = np.exp(-phi-V)
    DUstar /= DUstar.mean()
    method.compute_pull_back(push, phi, psi, DUstar)
    
    u = np.zeros((n,n))
    f = - push + mu
    solve_poisson(u,f,kernel,theta1,theta2)

    psi[:] = psi + u

    flt2d.find_c_concave(phi, psi, tau)

# %%

# %%
import tqdm

for ii in range(100,200):
    tau = 0.001
    sigma = 0.1

    mu = np.loadtxt(f'grf/grf-{ii}.csv', delimiter=',').astype('float64')
    mu /= mu.mean()
    print(mu.min(),mu.max(),mu.mean())
    n = mu.shape[0]
    method = bfmgf.BFM(n,n,tau)
    flt2d  = bfmgf.FLT2D(n,n)

    xx, yy = np.meshgrid(np.linspace(0.5/n,1-0.5/n,n),np.linspace(0.5/n,1-0.5/n,n))
    kernel = initialize_kernel(n, n, 1.0/n)

    

    DUstar = np.ones((n,n)).astype('float64')
    V      = ((xx-0.5)**2 + (yy-0.5)**2)/2.0 * 0.1
    V      = V.astype('float64')

    phi = np.zeros((n,n)).astype('float64')
    psi = np.zeros((n,n)).astype('float64')
    push   = np.zeros((n,n)).astype('float64')

    for jj in range(2   0):
        print('jj',jj)
        for i in range(1000):
            theta1 = 0.1
            theta2 = 0.1/(tau*np.max(mu))
            iterate_forward( bfmgf, flt2d, method, push, psi, phi, mu, DUstar, V, kernel, n, tau, theta1, theta2)
            iterate_backward(bfmgf, flt2d, method, push, psi, phi, mu, DUstar, V, kernel, n, tau, theta1, theta2)
            
        # bfmgf.calculate_DUstar(DUstar, V, phi, n, n, tau)
        DUstar = np.exp(-phi - V)
        DUstar /= DUstar.mean()

        # if ii % 50 == 0 and jj == 0:
        fig,ax=plt.subplots(1,3)
        ax[0].contourf(xx,yy,mu,10)
        ax[0].set_aspect('equal')
        ax[0].set_title(f"grf-{ii}")

        ax[1].contourf(xx,yy,phi,10)
        ax[1].set_aspect('equal')
        ax[1].set_title(f"phi-{ii}-{jj}")

        ax[2].contourf(xx,yy,DUstar,10)
        ax[2].set_aspect('equal')
        ax[2].set_title(f"rho-{ii}-{jj}")
        plt.savefig(f'figures/rho_{ii}_{jj}.png')
        plt.close('all')

        mu = DUstar

        np.savetxt(f"data/phi-{ii}-n-{jj}.csv", phi, delimiter=',')
        np.savetxt(f"data/psi-{ii}-n-{jj}.csv", psi, delimiter=',')
        np.savetxt(f"data/rho-{ii}-n-{jj}.csv", mu, delimiter=',')