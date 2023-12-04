import numpy as np
from GRF import *
from utils import *
from BFM_alg import BFM
from scipy.interpolate import CubicSpline
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

if __name__ == "__main__":
    # define config
    Ns = args.ns
    Nx = args.nx

    Nt = args.nt
    dt = args.dt

    alpha = args.alp
    tau = args.tau
    potential_name = args.potential

    # define grid
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
        tmp_rho = grf_rho[i, ...]
        tmp_rho = interp_2d(points_x, tmp_rho, points_xn, lx)

        rho0_mat[i, ...] = tmp_rho/tmp_rho.mean()

    if potential_name == 'gaussian':
        V = ((xxn-0.5)**2 + (yyn-0.5)**2)/2.0 * 0.1

    ### run BFM ####
    rho_evo_mat = np.zeros((Ns, Nt, Nx, Nx))

    for i in range(Ns):
        tmp_traj = BFM(dt, Nt, Nx, rho0_mat[i, ...], V)
        rho_evo_mat[i, ...] = tmp_traj

    ### save file
    npy_name = 'Kinetic_2D_' + potential_name + '_Ns_' + str(Ns) + '_Nx_' + str(Nx) + '_Nt_'+num2str_deciaml(Nt) + '_dt_' + num2str_deciaml(dt) + '_alp_' + num2str_deciaml(alpha) + '_tau_'+num2str_deciaml(tau) + '.npy'
    with open(npy_name, 'wb') as ss:
        np.save(ss, rho_evo_mat)












