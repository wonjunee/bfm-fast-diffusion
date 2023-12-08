import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from FNO_Net import FNODEQ, FNO2d
from GF_utils import *
import argparse
import time
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Operator learning of WGF')

parser.add_argument('-p', '--potential', type=str, metavar='', help='name of potential')
parser.add_argument('-nst', '--nst', type=int, metavar='', help='number of samples')
parser.add_argument('-nx', '--nx', type=int, metavar='', help='num of grids')
parser.add_argument('-nt', '--nt', type=int, metavar='', help='num of step')
parser.add_argument('-dt', '--dt', type=float, metavar='', help='time step size')
parser.add_argument('-bs', '--bs', type=int, metavar='', help='batch size')
parser.add_argument('-alp', '--alp', type=float, metavar='', help='alpha in GRF')
parser.add_argument('-tau', '--tau', type=int, metavar='', help='tau in GRF')

parser.add_argument('-n', '--net', type=str, metavar='', help='network architecture')
parser.add_argument('-st', '--st', type=int, metavar='', help='number of steps')
parser.add_argument('-flg', '--flg', type=int, metavar='', help='flag indicates rho or phi learning')
parser.add_argument('-inj', '--inj', type=str, metavar='', help='x injection')

parser.add_argument('-md', '--modes', type=int, metavar='', required=False, help='modes in FNO layer')
parser.add_argument('-tol', '--tol', type=float, metavar='', required=False, help='tolerence in AA fixed point')
parser.add_argument('-mi', '--maxiter', type=int, metavar='', required=False, help='max iterations in AA fixed point')
parser.add_argument('-nr', '--nr', type=int, metavar='', required=False, help='num of neurons')

args = parser.parse_args()

if __name__ == "__main__":
    # define config
    args = None
    if args:
        # config from parser
        Ns = args.nst
        Nx = args.nx
        bs = args.bs
        Nt = args.nt
        dt = args.dt
        alpha = args.alp
        tau = args.tau
        potential_name = args.potential

        net = args.net
        st = args.st
        flag = args.flg
        inj = args.inj

        tol = args.tol if args.tol != 'None' else None
        modes = args.modes if args.modes != 'None' else None
        maxiter = args.maxiter if args.maxiter != 'None' else None
        nr = args.nr if args.nr != 'None' else None


    else:
        # define here
        #net = 'DEQ2'
        #net = 'UNet'
        #net = 'FNO'
        net = 'FNO'

        inj = 'id'

        st = 2
        flag = 1

        modes = 12
        maxiter = 25
        nr = 16
        tol = 0.01


    # prepare data set
    lx = 1

    points_x = np.linspace(0.5 / Nx, lx - 0.5 / Nx, Nx)
    x = points_x[:, None]
    xx, yy = np.meshgrid(x, x)

    npy_name = 'Kinetic_2D_' + potential_name + '_Ns_' + str(Ns) + '_Nx_' + str(Nx) + '_Nt_' + num2str_deciaml(
        Nt) + '_dt_' + num2str_deciaml(dt) + '_alp_' + num2str_deciaml(alpha) + '_tau_' + num2str_deciaml(
        tau) + '_bs_' + num2str_deciaml(bs) + '_idx_' + num2str_deciaml(idx) + '.npy'

    with open(npy_name, 'rb') as ss:
        rho0_mat = np.load(ss)
        rho_mat = np.load(ss)
        phi_mat = np.load(ss)

    if rnd:
        Ns, Nt, Nk = 800, 50, 10
    else:
        Ns, Nt, Nk = Train_IP.shape[0], Test_IP.shape[0], Test_Traj.shape[0]

    ### load only partial data to train
    if rnd:
        print('randomly pick')
        train_idx = np.random.randint(0, Train_IP.shape[0], Ns)
        test_idx = np.random.randint(0, Test_IP.shape[0], Nt)
        traj_idx = np.random.randint(0, Test_rho0.shape[0], Nk)

        Train_IP, Train_OP, Test_IP, Test_OP, Test_rho0, Test_Traj = nump2tensor(Train_IP[train_idx, None, :, :]).to(
            device), nump2tensor(Train_OP[train_idx, None, :, :]).to(device), nump2tensor(
            Test_IP[test_idx, None, :, :]).to(device), nump2tensor(Test_OP[test_idx, None, :, :]).to(
            device), nump2tensor(Test_rho0[traj_idx, None, :, :]).to(device), nump2tensor(
            Test_Traj[traj_idx, :, None, :, :]).to(device)
    else:
        print('all use')
        Train_IP, Train_OP, Test_IP, Test_OP, Test_rho0, Test_Traj = nump2tensor(Train_IP[:, None, :, :]).to(device), nump2tensor(Train_OP[:, None, :, :]).to(device), nump2tensor(Test_IP[:, None, :, :]).to(device), nump2tensor(Test_OP[:, None, :, :]).to(device), nump2tensor(Test_rho0[:, None, :, :]).to(device), nump2tensor(Test_Traj[:, :, None, :, :]).to(device)

    print(Train_IP.shape, Train_OP.shape, Test_IP.shape, Test_OP.shape, Test_rho0.shape, Test_Traj.shape)
    epoches = 100000
    bs = int(Ns / 20) if Ns>400 else 10

    ### define model and hyperparameters
    common_name = '_st_' + str(st) + '_inj_' + inj + '_flg_' + str(flag)

    if net == 'DEQ':
        model = DEQ(16, 16, 32, inj, tol=tol, max_iter=maxiter, m=5).to(device)
        filename = net + common_name + '_maxiter_' + num2str_deciaml(maxiter) + '_tol_' + num2str_deciaml(tol) + '_Nj_' + num2str_deciaml(Nj) + '_bs_' + num2str_deciaml(bs)
    elif net == 'FNODEQ':
        model = FNODEQ(modes, modes, nr, inj, tol=tol, max_iter=maxiter, m=5).to(device)
        filename = net + common_name + '_mode_' + num2str_deciaml(modes) + '_nr_' + num2str_deciaml(nr) + '_maxiter_' + num2str_deciaml(maxiter) + '_tol_' + num2str_deciaml(tol) + '_Nj_' + num2str_deciaml(Nj) + '_bs_' + num2str_deciaml(bs)
    elif net == 'FNO':
        model = FNO2d(modes, modes, nr, 1, 1).to(device)
        filename = net + common_name + '_mode_' + num2str_deciaml(modes) + '_nr_' + num2str_deciaml(
            nr) + '_Nj_' + num2str_deciaml(Nj) + '_bs_' + num2str_deciaml(bs)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print(total_params)
    # zxc

    opt = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=2000, gamma=0.95)

    cwd = os.getcwd()
    model_name = cwd + '/mdls/' + filename + '.pt'
    log_name = filename + '_log.txt'
    fig_name = cwd + '/figs/' + filename

    content = 'The net is %s with total parameters is %d and training batchsize is %d' % (net, total_params, bs)
    mylogger(log_name, content)

    loss_func = nn.MSELoss()

    ### start training
    tic = time.time()
    for i in range(epoches + 1):
        tmp_IP, tmp_OP = get_batch(Ns, bs, Train_IP, Train_OP)
        tmp_IP.requires_grad = True


        tmp_pd = model(tmp_IP)

        loss = loss_func(tmp_pd, tmp_OP)
        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()

        if i % 100 == 0 and i > 0:

            toc = time.time() - tic
            total_loss = get_total_loss(Ns, 10, model, loss_func, Train_IP, Train_OP)

            test_loss = get_total_loss(Nt, 10, model, loss_func, Test_IP, Test_OP)

            content = 'at step %d the total training time is %3f the train loss is: %3f and the test loss is: %3f' % (
            i, toc, total_loss, test_loss)
            mylogger(log_name, content)
            print(content)

            get_plot(model, Test_IP, Test_OP, i, xx, yy, filename, fig_name)
            get_plot_evo(model, Test_rho0, Test_Traj, ast, Nx, i, xx, yy, filename, fig_name, cc, flag)

    torch.save(model.state_dict(), model_name)






