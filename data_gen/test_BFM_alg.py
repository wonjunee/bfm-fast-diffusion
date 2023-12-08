import numpy as np
import matplotlib.pyplot as plt
from utils import *
from GRF import *
from scipy.fftpack import dctn, idctn
from BFM_alg import BFM
from Potential import Potential
import bfmgf

#####################
import argparse

parser = argparse.ArgumentParser(description='data generation for 2D kinetic')

parser.add_argument('-p', '--potential', type=str, metavar='', help='name of potential')

parser.add_argument('-ns', '--ns', type=int, metavar='', help='number of samples')
parser.add_argument('-nx', '--nx', type=int, metavar='', help='num of grids')
parser.add_argument('-nt', '--nt', type=int, metavar='', help='num of step')
parser.add_argument('-dt', '--dt', type=float, metavar='', help='time step size')

parser.add_argument('-alp', '--alp', type=float, metavar='', help='alpha in GRF')
parser.add_argument('-tau', '--tau', type=int, metavar='', help='tau in GRF')

args = parser.parse_args()

Ns = args.ns
Nx = args.nx

Nt = args.nt
dt = args.dt

alpha = args.alp
tau = args.tau
potential_name = args.potential

#############################################

# Ns = 2
# Nx = 64
#
# Nt = 40
# dt = 0.01
#
# alpha = 4.0
# tau = 10
# potential_name = 'gaussian'

# define grid
lx = 1
points_x = np.linspace(0, lx - lx/Nx, Nx)
x = points_x[:, None]
xx, yy = np.meshgrid(x, x)

points_xn = np.linspace(0.5/Nx, lx-0.5/Nx,Nx)
xn = points_xn[:, None]
xxn, yyn = np.meshgrid(xn, xn)

GRF = GaussianRF(2, Nx, alpha, tau)

grf_rho = GRF.sample(1)
rho0 = grf_rho[0, ...]

rho0 = (torch.nn.functional.softplus(rho0, 1)).numpy()
mu = rho0 / rho0.mean()


### ----- No need to modfy --------
# Initialize Fourier kernel
def initialize_kernel(n1, n2, dy):
    xx, yy = np.meshgrid(np.linspace(0, np.pi, n1, False), np.linspace(0, np.pi, n2, False))
    # kernel = 2*n1*n1*(1-np.cos(xx)) + 2*n2*n2*(1-np.cos(yy))
    kernel = 2 * (1 - np.cos(xx)) / (dy * dy) + 2 * (1 - np.cos(yy)) / (dy * dy)
    kernel[0, 0] = 1  # to avoid dividing by zero
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
def solve_poisson(u, f, kernel, theta1, theta2):
    n = u.shape[0]
    u[:] = 0
    workspace = np.copy(f)
    workspace[0, 0] = 1
    workspace = dct2(workspace) / (theta1 + theta2 * kernel)
    workspace[0, 0] = 0
    u += idct2(workspace)


# %%


def iterate_forward(flt2d, method, push, psi, phi, mu, DUstar, V, kernel, n, tau, theta1, theta2):
    flt2d.find_c_concave(psi, phi, tau)
    flt2d.find_c_concave(phi, psi, tau)

    # bfmgf.calculate_DUstar(DUstar, V, phi, n, n, tau)
    DUstar = np.exp(-phi - V)
    DUstar /= DUstar.mean()

    method.compute_push_forth(push, phi, psi, mu)

    u = np.zeros((n, n))
    f = - push + DUstar
    solve_poisson(u, f, kernel, theta1, theta2)

    phi += u

    return np.mean(np.abs(u * f))


def iterate_backward(flt2d, method, push, psi, phi, mu, DUstar, V, kernel, n, tau, theta1, theta2):
    flt2d.find_c_concave(psi, phi, tau)

    # bfmgf.calculate_DUstar(DUstar, V, phi, n, n, tau)
    DUstar = np.exp(-phi - V)
    DUstar /= DUstar.mean()
    method.compute_pull_back(push, phi, psi, DUstar)

    u = np.zeros((n, n))
    f = - push + mu
    solve_poisson(u, f, kernel, theta1, theta2)

    psi += u

    flt2d.find_c_concave(phi, psi, tau)
    return np.mean(np.abs(u * f))


### ----- No need to modfy --------
n = Nx

xx, yy = np.meshgrid(np.linspace(0.5/n,1-0.5/n,n),np.linspace(0.5/n,1-0.5/n,n))
P = Potential(xx, yy)

if potential_name == 'double_wells':
    V = P.double_well(0.25, 0.25, 0.75, 0.75, 100, 0.01)
elif potential_name == 'gaussian':
    V = P.gaussian(0.5, 0.5, 1)
elif potential_name == 'trig':
    V = P.trig(3, 3, 1)


# for the BFM.
method = bfmgf.BFM(n,n,tau)
flt2d  = bfmgf.FLT2D(n,n)
kernel = initialize_kernel(n, n, 1.0/n)
DUstar = np.ones((n,n)).astype('float64')
V      = V.astype('float64')
phi = np.zeros((n,n)).astype('float64')
psi = np.zeros((n,n)).astype('float64')
push   = np.zeros((n,n)).astype('float64')


rho_traj, phi_traj = BFM(dt, Nt, Nx, mu, V)

eq = np.exp(-V)
eq = eq/eq.mean()



fig1, ax = plt.subplots(3, 2)
cp = ax[0, 0].contourf(xx, yy, rho_traj[0, ...], 10)
ax[0, 0].set_title(r'$\rho_0$')
plt.colorbar(cp)

cp = ax[1, 0].contourf(xx, yy, phi_traj[0, ...], 10)
ax[1, 0].set_title(r'$\phi_0$')
plt.colorbar(cp)

cp = ax[0, 1].contourf(xx, yy, rho_traj[-1, ...], 10)
ax[0, 1].set_title(r'$\rho$ end')
plt.colorbar(cp)

cp = ax[1, 1].contourf(xx, yy, phi_traj[-1, ...], 10)
ax[1, 1].set_title(r'$\phi$ end')
plt.colorbar(cp)

cp = ax[2, 0].contourf(xx, yy, eq, 10)
ax[2, 0].set_title(r'$eq$')
plt.colorbar(cp)
cp = ax[2, 1].contourf(xx, yy, np.abs(eq - rho_traj[-1, ...]), 10)
ax[2, 1].set_title(r'error')
plt.colorbar(cp)
plt.tight_layout()
fig1.savefig(f'end.png')


fig, ax = plt.subplots(2, 11, figsize=(20, 4))
cp = ax[0, 0].contourf(xx, yy, mu, 10)
ax[0, 0].set_title(r'$\rho_0$')

cp = ax[1, 0].contourf(xx, yy, V, 10)
ax[1, 0].set_title(r'$V$')

for i in range(10):
    cp = ax[0, i+1].contourf(xx, yy, rho_traj[int(Nt/10)*i, ...], 10)
    ax[0, i+1].set_title(r'$\rho$' + f"-{int(Nt/10)*i}")

    ax[1, i + 1].contourf(xx, yy, phi_traj[int(Nt / 10) * i, ...], 10)
    ax[1, i + 1].set_title(r'$\phi$' + f"-{int(Nt / 10) * i}")

for axs in ax.flat:
    axs.set_xticks([])
    axs.set_yticks([])

plt.tight_layout()
plt.savefig(f'all.png')
plt.close('all')