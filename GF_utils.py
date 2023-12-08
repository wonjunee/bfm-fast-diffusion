import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, 0, 0.01)

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

def nump2tensor(x):
    return torch.from_numpy(x).float()

def mylogger(filename, content):
    with open(filename, 'a') as fw:
        print(content, file=fw)

def get_batch(Ns, bs, rho, psi):
    idx = torch.randperm(Ns)[:bs]
    tmp_rho, tmp_psi = rho[idx], psi[idx]
    return tmp_rho.to(device), tmp_psi.to(device)

def get_total_loss(Ns, step, model, loss_func, IP, OP):
    loss = 0
    num = int(Ns/step)
    for i in range(step):
        with torch.no_grad():
            #print(i, num)
            tmp_IP, tmp_OP = IP[i*num:(i+1)*num, ...].to(device), OP[i*num:(i+1)*num, ...].to(device)
            tmp_IP.requires_grad_(False)
            pd = model(tmp_IP)
            loss += loss_func(tmp_OP, pd)*num
            del tmp_IP, tmp_OP, pd
            torch.cuda.empty_cache()
    return loss/Ns

### predict trajectory iteratively
def get_pd_traj(model, st, Nx, xx, yy, rho0, cc, flag=1):
    # flag indicates whether the pred is phi or rho
    # flag = 1 means rho
    #flag = 2 means phi
    pd = np.zeros((st, Nx, Nx))
    rho = rho0[None, ...]
    for i in range(st):
        pd_tmp = model(rho)
        if flag == 1:
            rho = pd_tmp
            pd[i, ...] = pd_tmp.detach().cpu().numpy()[0, 0, ...]
        elif flag == 2:
            rho_np = np.exp(-pd_tmp.detach().cpu().numpy()[0, 0, ...]/cc - 0.05 * (xx ** 2 + yy ** 2))
            rho = nump2tensor(rho_np).to(device)[None, None, ...]
            pd[i, ...] = rho_np

    return pd

def get_traj_all(model, st, Nx, xx, yy, Test_rho0, Test_Traj, cc, flag):
    Traj = np.zeros_like(Test_Traj.detach().cpu().numpy())

    for i in range(Test_Traj.shape[0]):
        #rho0 = nump2tensor(Test_rho0[i, ...]).to(device)
        rho0 = Test_rho0[i, ...]
        pd = get_pd_traj(model, st-1, Nx, xx, yy, rho0, cc, flag)
        Traj[i, ...] = pd[:, None, ...]

    return Traj

def get_plot(model, Test_IP, Test_OP, step, xx, yy, filename, fig_name):
    #with torch.no_grad():
    Test_OP_pd = model(Test_IP)
    Test_OP_pd = Test_OP_pd.detach().cpu().numpy()

    idx = 0

    plot_name = filename + ' at step ' + str(step)

    fig1 = plt.figure(1)
    ax = fig1.add_subplot(111)
    cp = ax.contourf(xx, yy, Test_OP_pd[idx, 0, ...], levels=36)
    plt.title(plot_name + ' pred')
    plt.colorbar(cp)
    fig1.savefig(fig_name + '_step_' + str(step) + '_pd.jpg')

    fig2 = plt.figure(2)
    ax = fig2.add_subplot(111)
    cp = ax.contourf(xx, yy, Test_OP.detach().cpu().numpy()[idx, 0, ...], levels=36)
    plt.title(plot_name + ' ref')
    plt.colorbar(cp)
    fig2.savefig(fig_name + '_step_' + str(step) + '_ref.jpg')

    fig3 = plt.figure(3)
    ax = fig3.add_subplot(111)
    cp = ax.contourf(xx, yy, np.abs(Test_OP.detach().cpu().numpy()[idx, 0, ...] - Test_OP_pd[idx, 0, ...]), levels=36)
    plt.title(plot_name + ' abs error')
    plt.colorbar(cp)
    fig3.savefig(fig_name + '_step_' + str(step) + '_err.jpg')

    fig4 = plt.figure(4)
    ax = fig4.add_subplot(111)
    cp = ax.contourf(xx, yy, Test_IP.detach().cpu().numpy()[idx, 0, ...], levels=36)
    plt.title(plot_name + ' before')
    plt.colorbar(cp)
    fig4.savefig(fig_name + '_step_' + str(step) + '_bf.jpg')

    #plt.show()

    plt.close('all')


def get_plot_evo(model, Test_rho0, Test_Traj, st, Nx, step, xx, yy, filename, fig_name, cc, flag):

    Traj_pd = get_traj_all(model, st, Nx, xx, yy, Test_rho0, Test_Traj, cc, flag)

    avg_loss_in_time = np.sqrt(np.mean((Traj_pd - Test_Traj.detach().cpu().numpy())**2, axis=(0, 2, 3, 4)) / np.mean((Test_Traj.detach().cpu().numpy())**2, axis=(0, 2, 3, 4)))

    t_vec = np.linspace(1, st-1, st-1)

    idx = -2

    plot_name = filename + '\n' + ' evo end at step ' + str(step)

    # print(Traj_pd.shape)
    # zxc

    fig1 = plt.figure(1)
    ax = fig1.add_subplot(111)
    cp = ax.contourf(xx, yy, Traj_pd[idx, -1, 0, ...], levels=36)
    plt.title(plot_name + ' pred')
    plt.colorbar(cp)
    fig1.savefig(fig_name + '_evo_step_' + str(step) + '_pd.jpg')

    fig2 = plt.figure(2)
    ax = fig2.add_subplot(111)
    cp = ax.contourf(xx, yy, Test_Traj.detach().cpu().numpy()[idx, -1, 0, ...], levels=36)
    plt.title(plot_name + ' ref')
    plt.colorbar(cp)
    fig2.savefig(fig_name + '_evo_step_' + str(step) + '_ref.jpg')

    fig3 = plt.figure(3)
    ax = fig3.add_subplot(111)
    cp = ax.contourf(xx, yy, np.abs(Test_Traj.detach().cpu().numpy()[idx, -1, 0, ...] - Traj_pd[idx, -1, 0, ...]), levels=36)
    plt.title(plot_name + ' abs error')
    plt.colorbar(cp)
    fig3.savefig(fig_name + '_evo_step_' + str(step) + '_err.jpg')

    fig4 = plt.figure(4)
    plt.plot(t_vec, avg_loss_in_time, 'r-o')
    plt.title(plot_name + ' avg rl2 error')
    fig4.savefig(fig_name + '_evo_step_' + str(step) + '_rl2err.jpg')

    #plt.show()

    plt.close('all')
