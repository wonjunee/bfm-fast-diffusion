import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import numpy as np

def num2str_deciaml(x):
    s = str(x)
    c = ''
    for i in range(len(s)):
        if s[i] == '0':
            c = c + 'z'
        elif s[i] == '.':
            c = c + 'p'
        elif s[i] == '-':
            c = c + 'n'
        else:
            c = c + s[i]

    return c

def interp_1d(x, f, xnew, lx):
    x = np.concatenate([x, lx * np.ones((1))], axis=0)
    f = np.concatenate([f, f[0]*np.ones((1))], axis=0)
    func = CubicSpline(x, f)
    fnew = func(xnew)
    return fnew

def interp_2d(x, f, xnew, lx):
    N1 = x.shape[0]
    N2 = xnew.shape[0]

    fmid = np.zeros((N1, N2))
    fnew = np.zeros((N2, N2))

    for i in range(N1):
        fmid[i, :] = interp_1d(x, f[i, :], xnew, lx)
    for j in range(N1):
        fnew[:, j] = interp_1d(x, fmid[:, j], xnew, lx)
    return fnew

def get_plot(xx, yy, mu, V, rho_traj, phi_traj, fig_name):
    Nt = rho_traj.shape[0]
    fig1, ax = plt.subplots(2, 2)
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
    plt.tight_layout()
    fig1.savefig(fig_name + '_end.png')

    # print(mu.shape)
    # zxc

    fig, ax = plt.subplots(2, 11, figsize=(20, 4))
    cp = ax[0, 0].contourf(xx, yy, mu, 10)
    ax[0, 0].set_title(r'$\rho_0$')

    cp = ax[1, 0].contourf(xx, yy, V, 10)
    ax[1, 0].set_title(r'$V$')

    for i in range(10):
        cp = ax[0, i + 1].contourf(xx, yy, rho_traj[int(Nt / 10) * i, ...], 10)
        ax[0, i + 1].set_title(r'$\rho$' + f"-{int(Nt / 10) * i}")

        ax[1, i + 1].contourf(xx, yy, phi_traj[int(Nt / 10) * i, ...], 10)
        ax[1, i + 1].set_title(r'$\phi$' + f"-{int(Nt / 10) * i}")

    for axs in ax.flat:
        axs.set_xticks([])
        axs.set_yticks([])

    plt.tight_layout()
    plt.show()
    plt.savefig(fig_name + '_all.png')
    plt.close('all')