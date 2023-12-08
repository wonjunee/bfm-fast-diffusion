import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.autograd as autograd
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### FNO layer
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FourierLayer(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FourierLayer, self).__init__()
        self.conv0 = SpectralConv2d(width, width, modes1, modes2)
        self.w0 = nn.Conv2d(width, width, 1)

    def forward(self, x):
        y1 = self.conv0(x)
        y2 = self.w0(x)
        y = y1 + y2
        y = F.gelu(y)
        out = y
        return out

class FNO2d(nn.Module):
    def __init__(self, nl, modes1, modes2, width):
        super(FNO2d, self).__init__()
        self.fc0 = nn.Linear(1, width)  # input channel is 3: (a(x, y), x, y)

        self.layers_ls = nn.ModuleList()

        for i in range(nl):
            self.layers_ls.append(FourierLayer(modes1, modes2, width))

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):

        x = x.permute(0, 2, 3, 1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        for layer in self.layers_ls:
            x = layer(x)

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)
        return x

class FNO2dx(nn.Module):
    def __init__(self, nl, modes1, modes2, width):
        super(FNO2dx, self).__init__()
        self.fc0 = nn.Linear(3, width)  # input channel is 3: (a(x, y), x, y)
        self.layers_ls = nn.ModuleList()

        for i in range(nl):
            self.layers_ls.append(FourierLayer(modes1, modes2, width))

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):

        x = x.permute(0, 2, 3, 1)
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        for layer in self.layers_ls:
            x = layer(x)

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

###### DEQ ###########
def anderson(f, x0, m=5, lam=1e-4, max_iter=50, tol=1e-2, beta=1.0):
    """ Anderson acceleration for fixed point iteration. """
    bsz, d, H, W = x0.shape
    X = torch.zeros(bsz, m, d * H * W, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, d * H * W, dtype=x0.dtype, device=x0.device)
    # print(x0.shape, (f(x0)).shape)
    # zxc
    X[:, 0], F[:, 0] = x0.reshape(bsz, -1), f(x0).reshape(bsz, -1)
    X[:, 1], F[:, 1] = F[:, 0], f(F[:, 0].reshape_as(x0)).reshape(bsz, -1)

    H = torch.zeros(bsz, m + 1, m + 1, dtype=x0.dtype, device=x0.device)
    H[:, 0, 1:] = H[:, 1:, 0] = 1
    y = torch.zeros(bsz, m + 1, 1, dtype=x0.dtype, device=x0.device)
    y[:, 0] = 1

    res = []
    for k in range(2, max_iter):
        n = min(k, m)
        G = F[:, :n] - X[:, :n]
        H[:, 1:n + 1, 1:n + 1] = torch.bmm(G, G.transpose(1, 2)) + lam * torch.eye(n, dtype=x0.dtype, device=x0.device)[
            None]
        #alpha = torch.solve(y[:, :n + 1], H[:, :n + 1, :n + 1])[0][:, 1:n + 1, 0]  # (bsz x n)
        alpha = torch.linalg.solve(H[:, :n + 1, :n + 1], y[:, :n + 1])[:, 1:n + 1, 0]

        X[:, k % m] = beta * (alpha[:, None] @ F[:, :n])[:, 0] + (1 - beta) * (alpha[:, None] @ X[:, :n])[:, 0]
        F[:, k % m] = f(X[:, k % m].reshape_as(x0)).reshape(bsz, -1)
        res.append((F[:, k % m] - X[:, k % m]).norm().item() / (1e-5 + F[:, k % m].norm().item()))
        if (res[-1] < tol):
            break
    return X[:, k % m].reshape_as(x0), res



### injection method

class LinearInjection(nn.Module):
    def __init__(self, dims, nr):
        super().__init__()
        self.P = nn.Linear(dims, nr)
        self.Q = nn.Linear(nr, dims)

        self.P.weight.data.normal_(0, 0.01)
        self.Q.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = torch.sigmoid(self.P(x))
        x = torch.sigmoid(self.Q(x))
        x = x.permute(0, 3, 1, 2)
        return x

class ConvInjection(nn.Module):
    def __init__(self, dims, nr, kernel_size=3, num_groups=2):
        super().__init__()
        self.conv = nn.Conv2d(dims, nr, kernel_size, padding=kernel_size // 2, bias=False)
        self.Q = nn.Linear(nr, dims)
        self.norm = nn.GroupNorm(num_groups, nr)

        self.conv.weight.data.normal_(0, 0.01)
        self.Q.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)
        x = torch.sigmoid(self.Q(x))
        x = x.permute(0, 3, 1, 2)
        return x

### fixed point layer
class FP_FNO_Id_Inj(nn.Module):
    def __init__(self, modes1, modes2, width, num_groups=2):
        super().__init__()
        self.FL = FourierLayer(modes1, modes2, width)
        self.norm1 = nn.GroupNorm(num_groups, width)
        self.norm2 = nn.GroupNorm(num_groups, width)

    def forward(self, z, x):
        y = self.FL(z)
        return self.norm2(F.relu(z + self.norm1(x + y)))

class FP_FNO_Linear_Inj(nn.Module):
    def __init__(self, modes1, modes2, width, nr=16, num_groups=2):
        super().__init__()
        self.FL = FourierLayer(modes1, modes2, width)
        self.norm1 = nn.GroupNorm(num_groups, width)
        self.norm2 = nn.GroupNorm(num_groups, width)
        self.Inj = LinearInjection(width, nr)

    def forward(self, z, x):
        x = self.Inj(x)
        y = self.FL(z)
        return self.norm2(F.relu(z + self.norm1(x + y)))

class FP_FNO_Conv_Inj(nn.Module):
    def __init__(self, modes1, modes2, width, nr=16, num_groups=2):
        super().__init__()
        self.FL = FourierLayer(modes1, modes2, width)
        self.norm1 = nn.GroupNorm(num_groups, width)
        self.norm2 = nn.GroupNorm(num_groups, width)
        self.Inj = ConvInjection(width, nr)

    def forward(self, z, x):
        x = self.Inj(x)
        y = self.FL(z)
        return self.norm2(F.relu(z + self.norm1(x + y)))

### DEQ
class DEQFixedPoint(nn.Module):
    def __init__(self, f, solver, **kwargs):
        super().__init__()
        self.f = f
        self.solver = solver
        self.kwargs = kwargs

    def forward(self, x):
        # compute forward pass and re-engage autograd tape
        with torch.no_grad():
            z, self.forward_res = self.solver(lambda z: self.f(z, x), torch.zeros_like(x), **self.kwargs)
        z = self.f(z, x)

        # set up Jacobian vector product (without additional forward calls)
        z0 = z.clone().detach().requires_grad_()
        f0 = self.f(z0, x)

        if not x.requires_grad:
            return z
        else:
            def backward_hook(grad):
                g, self.backward_res = self.solver(lambda y: autograd.grad(f0, z0, y, retain_graph=True)[0] + grad,
                                                   grad, **self.kwargs)
                return g

            z.register_hook(backward_hook)
            return z

class FNODEQ(nn.Module):
    def __init__(self, modes1, modes2, width, inj, **kwargs):
        super(FNODEQ, self).__init__()
        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.input_dims, self.out_dims = 1, 1
        self.padding = 1  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(self.input_dims, self.width)  # input channel is 3: (a(x, y), x, y)

        if inj == 'id':
            self.f = FP_FNO_Id_Inj(modes1, modes2, width)
        elif inj == 'lin':
            self.f = FP_FNO_Linear_Inj(modes1, modes2, width)
        elif inj == 'conv':
            self.f = FP_FNO_Conv_Inj(modes1, modes2, width)
        self.norm = nn.BatchNorm2d(self.width)
        self.FPL = DEQFixedPoint(self.f, anderson, **kwargs)
        self.fc1 = nn.Linear(self.width, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):

        x = x.permute(0, 2, 3, 1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x = self.norm(self.FPL(x))

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)
        return x

