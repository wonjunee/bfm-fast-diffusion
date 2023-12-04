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