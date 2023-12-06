import numpy as np
import matplotlib.pyplot as plt
from utils import *

Ns = 2
Nx = 64

Nt = 10
dt = 0.0005

alpha = 4.0
tau = 10
potential_name = 'gaussian'

# define grid
lx = 1
points_x = np.linspace(0, lx - lx/Nx, Nx)
x = points_x[:, None]
xx, yy = np.meshgrid(x, x)

points_xn = np.linspace(0.5/Nx, lx-0.5/Nx,Nx)
xn = points_xn[:, None]
xxn, yyn = np.meshgrid(xn, xn)


npy_name = 'Kinetic_2D_' + potential_name + '_Ns_' + str(Ns) + '_Nx_' + str(Nx) + '_Nt_'+num2str_deciaml(Nt) + '_dt_' + num2str_deciaml(dt) + '_alp_' + num2str_deciaml(alpha) + '_tau_'+num2str_deciaml(tau) + '.npy'
with open(npy_name, 'rb') as ss:
    rho0_mat = np.load(ss)
    rho_mat = np.load(ss)
    phi_mat = np.load(ss)


fig,ax=plt.subplots(1,3)
cp = ax[0].contourf(xx,yy,rho_mat[0, 0, ...],10)
ax[0].set_aspect('equal')
ax[0].set_title(r'$\rho$')
plt.colorbar(cp)

cp = ax[1].contourf(xx,yy,rho_mat[0, 1, ...],10)
ax[1].set_aspect('equal')
ax[1].set_title(r'$\rho$')
plt.colorbar(cp)

cp = ax[2].contourf(xx,yy,rho_mat[0, -1, ...],10)
ax[2].set_aspect('equal')
ax[2].set_title(r'$\rho$')
plt.colorbar(cp)

fig2,ax=plt.subplots(1,3)
cp = ax[0].contourf(xx,yy,phi_mat[0, 0, ...],10)
ax[0].set_aspect('equal')
ax[0].set_title(r'$\phi$')
plt.colorbar(cp)

cp = ax[1].contourf(xx,yy,phi_mat[0, 2, ...],10)
ax[1].set_aspect('equal')
ax[1].set_title(r'$\phi$')
plt.colorbar(cp)

cp = ax[2].contourf(xx,yy,phi_mat[0, -1, ...],10)
ax[2].set_aspect('equal')
ax[2].set_title(r'$\phi$')
plt.colorbar(cp)

plt.show()