import numpy as np
from GRF import *
from utils import *
from BFM_alg import BFM
from Potential import Potential
import argparse

parser = argparse.ArgumentParser(description='data generation for 2D kinetic')

parser.add_argument('-p', '--potential', type=str, metavar='', help='name of potential')

parser.add_argument('-nst', '--nst', type=int, metavar='', help='number of samples')
parser.add_argument('-nx', '--nx', type=int, metavar='', help='num of grids')
parser.add_argument('-nt', '--nt', type=int, metavar='', help='num of step')
parser.add_argument('-dt', '--dt', type=float, metavar='', help='time step size')

parser.add_argument('-bs', '--bs', type=int, metavar='', help='batch size')
parser.add_argument('-bid', '--bid', type=int, metavar='', help='batch index')

parser.add_argument('-alp', '--alp', type=float, metavar='', help='alpha in GRF')
parser.add_argument('-tau', '--tau', type=int, metavar='', help='tau in GRF')

args = parser.parse_args()

if __name__ == "__main__":
    # define config
    Ns = args.nst
    Nx = args.nx

    bs = args.bs
    bid = args.bid

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

    # load IC
    npy_name = 'IC_2D_Ns_' + str(Ns) + '_Nx_' + str(Nx) + '_alp_' + num2str_deciaml(alpha) + '_tau_'+num2str_deciaml(tau) + '.npy'
    with open(npy_name, 'rb') as ss:
        rho0_mat = np.load(ss)

    rho0_mat = rho0_mat[bid*bs:(bid+1)*bs, ...]

    P = Potential(xxn, yyn)
    if potential_name == 'double_wells':
        V = P.double_well(0.25, 0.25, 0.75, 0.75, 100, 1)
    elif potential_name == 'gaussian':
        V = P.gaussian(0.5, 0.5, 1)
    elif potential_name == 'trig':
        V = P.trig(3, 3, 1)


    ### run BFM ####
    rho_evo_mat = np.zeros((bs, Nt, Nx, Nx))
    phi_evo_mat = np.zeros((bs, Nt, Nx, Nx))

    for i in range(bs):
        tmp_rho_traj, tmp_phi_traj = BFM(dt, Nt, Nx, rho0_mat[i, ...], V)
        rho_evo_mat[i, ...] = tmp_rho_traj
        phi_evo_mat[i, ...] = tmp_phi_traj

    ### save file
    npy_name = 'Kinetic_2D_' + potential_name + '_Ns_' + str(Ns) + '_Nx_' + str(Nx) + '_Nt_'+num2str_deciaml(Nt) + '_dt_' + num2str_deciaml(dt) + '_alp_' + num2str_deciaml(alpha) + '_tau_'+num2str_deciaml(tau)+ '_bs_' + num2str_deciaml(bs) + '_idx_' + num2str_deciaml(bid) + '.npy'
    with open(npy_name, 'wb') as ss:
        np.save(ss, rho0_mat)
        np.save(ss, rho_evo_mat)
        np.save(ss, phi_evo_mat)












