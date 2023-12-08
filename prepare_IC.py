import numpy as np
from GRF import *
from utils import *

import argparse

parser = argparse.ArgumentParser(description='data generation for 2D kinetic')

parser.add_argument('-ns', '--ns', type=int, metavar='', help='number of samples')
parser.add_argument('-nx', '--nx', type=int, metavar='', help='num of grids')

parser.add_argument('-alp', '--alp', type=float, metavar='', help='alpha in GRF')
parser.add_argument('-tau', '--tau', type=int, metavar='', help='tau in GRF')

args = parser.parse_args()

Ns = args.ns
Nx = args.nx

alpha = args.alp
tau = args.tau

lx = 1
points_x = np.linspace(0, lx - lx/Nx, Nx)
x = points_x[:, None]
xx, yy = np.meshgrid(x, x)

points_xn = np.linspace(0.5/Nx, lx-0.5/Nx,Nx)
xn = points_xn[:, None]
xxn, yyn = np.meshgrid(xn, xn)

GRF = GaussianRF(2, Nx, alpha, tau)
rho0_mat = np.zeros((Ns, Nx, Nx))

grf_rho = GRF.sample(Ns)

for i in range(Ns):
    tmp_rho = (torch.nn.functional.softplus(grf_rho[i, ...], 1)).numpy()
    tmp_rho = interp_2d(points_x, tmp_rho, points_xn, lx)

    rho0_mat[i, ...] = tmp_rho/tmp_rho.mean()

npy_name = 'IC_2D_Ns_' + str(Ns) + '_Nx_' + str(Nx) + '_alp_' + num2str_deciaml(alpha) + '_tau_'+num2str_deciaml(tau) + '.npy'
with open(npy_name, 'wb') as ss:
    np.save(ss, rho0_mat)