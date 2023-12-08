import numpy as np
import matplotlib.pyplot as plt
from utils import *
from Potential import Potential

Ns = 20
bs = 2
idx = 1

Nx = 64

Nt = 200
dt = 0.0001

alpha = 4.0
tau = 10
potential_name = 'trig'



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
    V = P.double_well(0.25, 0.25, 0.75, 0.75, 100, 0.01)
elif potential_name == 'gaussian':
    V = P.gaussian(0.5, 0.5, 0.01)
elif potential_name == 'trig':
    V = P.trig(3, 3, 0.01)


#npy_name = 'Kinetic_2D_' + potential_name + '_Ns_' + str(Ns) + '_Nx_' + str(Nx) + '_Nt_'+num2str_deciaml(Nt) + '_dt_' + num2str_deciaml(dt) + '_alp_' + num2str_deciaml(alpha) + '_tau_'+num2str_deciaml(tau) + '.npy'
npy_name = 'Kinetic_2D_' + potential_name + '_Ns_' + str(Ns) + '_Nx_' + str(Nx) + '_Nt_'+num2str_deciaml(Nt) + '_dt_' + num2str_deciaml(dt) + '_alp_' + num2str_deciaml(alpha) + '_tau_'+num2str_deciaml(tau) +'_bs_' + num2str_deciaml(bs) + '_idx_' + num2str_deciaml(idx) + '.npy'

with open(npy_name, 'rb') as ss:
    rho0_mat = np.load(ss)
    rho_mat = np.load(ss)
    phi_mat = np.load(ss)

# print(rho0_mat.shape)
# zxc

get_plot(xxn, yyn, rho0_mat[0, ...], V, rho_mat[0, ...], phi_mat[0, ...], 'fig')

#
# fig,ax=plt.subplots(1,3)
# cp = ax[0].contourf(xx,yy,rho_mat[0, 0, ...],10)
# ax[0].set_aspect('equal')
# ax[0].set_title(r'$\rho$')
# plt.colorbar(cp)
#
# cp = ax[1].contourf(xx,yy,rho_mat[0, 1, ...],10)
# ax[1].set_aspect('equal')
# ax[1].set_title(r'$\rho$')
# plt.colorbar(cp)
#
# cp = ax[2].contourf(xx,yy,rho_mat[0, -1, ...],10)
# ax[2].set_aspect('equal')
# ax[2].set_title(r'$\rho$')
# plt.colorbar(cp)
#
# fig2,ax=plt.subplots(1,3)
# cp = ax[0].contourf(xx,yy,phi_mat[0, 0, ...],10)
# ax[0].set_aspect('equal')
# ax[0].set_title(r'$\phi$')
# plt.colorbar(cp)
#
# cp = ax[1].contourf(xx,yy,phi_mat[0, 2, ...],10)
# ax[1].set_aspect('equal')
# ax[1].set_title(r'$\phi$')
# plt.colorbar(cp)
#
# cp = ax[2].contourf(xx,yy,phi_mat[0, -1, ...],10)
# ax[2].set_aspect('equal')
# ax[2].set_title(r'$\phi$')
# plt.colorbar(cp)
#
# plt.show()