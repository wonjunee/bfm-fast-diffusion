import numpy as np
from GRF import *
from utils import *
from BFM_alg import BFM
from Potential import Potential
from data_gen_func import gen_batch
import multiprocessing
import functools
import argparse

parser = argparse.ArgumentParser(description='data generation for 2D kinetic')

parser.add_argument('-p', '--potential', type=str, metavar='', help='name of potential')
parser.add_argument('-nst', '--nst', type=int, metavar='', help='number of samples')
parser.add_argument('-nx', '--nx', type=int, metavar='', help='num of grids')
parser.add_argument('-nt', '--nt', type=int, metavar='', help='num of step')
parser.add_argument('-dt', '--dt', type=float, metavar='', help='time step size')
parser.add_argument('-bs', '--bs', type=int, metavar='', help='batch size')
parser.add_argument('-alp', '--alp', type=float, metavar='', help='alpha in GRF')
parser.add_argument('-tau', '--tau', type=int, metavar='', help='tau in GRF')

args = parser.parse_args()

if __name__ == "__main__":
    # define config
    Ns = args.nst
    Nx = args.nx

    bs = args.bs

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

    P = Potential(xxn, yyn)
    if potential_name == 'double_wells':
        V = P.double_well(0.25, 0.25, 0.75, 0.75, 100, 1)
    elif potential_name == 'gaussian':
        V = P.gaussian(0.5, 0.5, 1)
    elif potential_name == 'trig':
        V = P.trig(3, 3, 1)

    ### define partial func
    gen_f = functools.partial(gen_batch, Ns, bs, Nx, Nt, dt, alpha, tau, potential_name)
    index_ls = [i for i in range(int(Ns/bs))]

    with multiprocessing.Pool() as pool:
        results = pool.map(gen_f, index_ls)












