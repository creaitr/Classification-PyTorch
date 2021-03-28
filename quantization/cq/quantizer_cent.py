import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad



class CQ(nn.Module):
    def __init__(self, nbits, q_n, q_p, weight=None):
        super().__init__()
        self.nbits = Parameter(torch.Tensor(1), requires_grad=False)
        self.nbits.fill_(nbits)

        self.q_n = q_n
        self.q_p = q_p
        self.t_n = -1.
        self.t_p = 1.
        
        self.step = Parameter(torch.Tensor(1))
        self.register_buffer('do_init', torch.zeros(1))
        self.mask = Parameter(torch.ones(weight.size()), requires_grad=False)

    @property
    def step_abs(self):
        return self.step.abs()

    def init_step(self, x, *args, **kwargs):
        self.step.data.copy_(
            2. * x.abs().mean() / math.sqrt(self.t_p)
        )
        self.do_init.fill_(1)

    def quantize(self, x, s, q_n, q_p, q):
        step_grad_scale = 1.0 / ((q * x.numel()) ** 0.5)
        step_scale = grad_scale(s, step_grad_scale)

        x = x / step_scale
        x = torch.clamp(x, q_n, q_p)
        x = round_pass(x)
        x = x * step_scale
        return x

    def forward(self, x):
        if self.training and self.do_init == 0:
            self.init_step(x)

        x_slw = self.quantize(x, self.step_abs, self.q_n, self.q_p, self.q_p)
        x_tw = self.quantize(x, self.step_abs, self.t_n, self.t_p, self.t_p)
        x = torch.where(self.mask == 1., x_tw, x_slw)
        
        return x

    def extra_repr(self):
        return 'nbits={}, q_n={}, q_p={}, t_n={}, t_p={}'.format(int(self.nbits[0]), int(self.q_n), int(self.q_p), int(self.t_n), int(self.t_p))
