import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dctn, idctn
import bfmgf

######## help func ###################################################################
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
##################################################################################

### main algorithm starts here ######################################
def BFM(tau, Nt, n, rho0, V, dtype= 'float64', maxiter=1000, tol=2e-3):
    #########
    # outer loop: tau, Nt, time T = Nt*tau
    # inner loop: sigma: lr; maxiter and tol
    # rho is IC
    # V is potential
    ###########
    sigma = tau/2

    mu = rho0.copy()

    rho_traj = np.zeros((Nt, n, n)).astype(dtype)
    phi_traj = np.zeros((Nt, n, n)).astype(dtype)

    method = bfmgf.BFM(n, n, tau)
    flt2d = bfmgf.FLT2D(n, n)
    kernel = initialize_kernel(n, n, 1.0 / n)
    DUstar = np.ones((n, n)).astype(dtype)
    V = V.astype(dtype)
    phi = np.zeros((n, n)).astype(dtype)
    psi = np.zeros((n, n)).astype(dtype)
    push = np.zeros((n, n)).astype(dtype)

    for t in range(Nt):  # total number of outer iterations rho^0, rho^1, ..., rho^{jj}
        error_for_prev = 1
        error_bac_prev = 1
        for i in range(maxiter):  # you may need samller numbero fiteration than this.
            theta1 = sigma
            theta2 = sigma / (tau * np.max(mu))
            error_for = iterate_forward(flt2d, method, push, psi, phi, mu, DUstar, V, kernel, n, tau, theta1, theta2)
            error_bac = iterate_backward(flt2d, method, push, psi, phi, mu, DUstar, V, kernel, n, tau, theta1, theta2)

            error1 = np.abs((error_for - error_for_prev) / error_for_prev)
            error2 = np.abs((error_bac - error_bac_prev) / error_bac_prev)
            error = min(error1, error2)

            #print(f'jj: {jj}  i: {i}  error: {error:0.2e}, {error_for:0.2e}, {error_bac:0.2e}')

            if error < tol:  # stopping condition
                break

            error_for_prev = error_for
            error_bac_prev = error_bac
        DUstar = np.exp(-phi - V)
        DUstar /= DUstar.mean()
        #### Wuzhe: is this step really needed? If so, the mapping from rho to phi can be different each time step, right?
        mu = DUstar

        rho_traj[t, ...] = mu.copy()
        phi_traj[t, ...] = phi.copy()

    return rho_traj, phi_traj
