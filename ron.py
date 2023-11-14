from esn import spectral_norm_scaling
import torch
import torch.nn as nn
from avalanche.models import IncrementalClassifier


class RON(nn.Module):
    """
    Batch-first (B, L, I)
    """
    def __init__(self, n_inp, n_hid, dt, gamma_min, gamma_max, eps_min, eps_max, rho, input_scaling, device='cpu',
                 initial_out_features=2, return_sequences=False):
        super().__init__()
        self.n_hid = n_hid
        self.input_size = n_inp
        self.device = device
        self.return_sequences = return_sequences
        self.dt = dt
        self.gamma = torch.rand(n_hid, requires_grad=False, device=device) * (gamma_max - gamma_min) + gamma_min
        self.epsilon = torch.rand(n_hid, requires_grad=False, device=device) * (eps_max - eps_min) + eps_min

        h2h = 2 * (2 * torch.rand(n_hid, n_hid) - 1)
        h2h = spectral_norm_scaling(h2h, rho)
        self.h2h = nn.Parameter(h2h, requires_grad=False)

        x2h = 2 * (2 * torch.rand(n_inp, self.n_hid) - 1) * input_scaling
        self.x2h = nn.Parameter(x2h, requires_grad=False)
        bias = (torch.rand(n_hid) * 2 - 1) * input_scaling
        self.bias = nn.Parameter(bias, requires_grad=False)

        self.readout = IncrementalClassifier(in_features=n_hid,
                                             initial_out_features=initial_out_features)

    def cell(self, x, hy, hz):
        hz = hz + self.dt * (torch.tanh(
            torch.matmul(x, self.x2h) + torch.matmul(hy, self.h2h) + self.bias)
                             - self.gamma * hy - self.epsilon * hz)
        hy = hy + self.dt * hz
        return hy, hz

    def forward(self, x):
        if len(x.shape) == 4:  # CIFAR, put channels together
            x = torch.transpose(x, 1, 3).contiguous()
        x = x.view(x.size(0), -1, self.input_size)  # (B, L, I)
        hy = torch.zeros(x.size(0), self.n_hid).to(self.device)
        hz = torch.zeros(x.size(0), self.n_hid).to(self.device)
        all_states = []
        for t in range(x.size(1)):
            hy, hz = self.cell(x[:, t], hy, hz)
            all_states.append(hy)
        all_states = torch.stack(all_states, dim=1)
        if self.return_sequences:
            out = self.readout(all_states)
        else:
            out = self.readout(all_states[:, -1])
        return out

