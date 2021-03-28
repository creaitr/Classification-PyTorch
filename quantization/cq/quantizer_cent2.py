import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class Inferer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, s, mask, q_n, q_p, t_n, t_p):
        x_n = x / s
        
        tw = x_n.clamp(t_n, t_p)
        slw = x_n.clamp(q_n, q_p)
        
        x_q = torch.where(mask == 1., tw, slw).round() * s
        
        ctx.save_for_backward(x_n, x_q, mask)
        ctx.other = q_n, q_p, t_n, t_p
        return x_q

    @staticmethod
    def backward(ctx, grad_x):
        x_n, x_q, mask = ctx.saved_tensors
        q_n, q_p, t_n, t_p = ctx.other
        
        gs_slw = math.sqrt(x_n.numel() * q_p)
        gs_tw = math.sqrt(x_n.numel() * t_p)
        
        # gradient of slw
        idx_s_slw = (x_n <= q_n).float()
        idx_l_slw = (x_n >= q_p).float()
        idx_m_slw = torch.ones(size=idx_s_slw.shape, device=idx_s_slw.device) - idx_s_slw - idx_l_slw
        
        grad_s_slw = (idx_s_slw * q_n + idx_l_slw * q_p + idx_m_slw * (-x_n + x_q)) * grad_x / gs_slw
        grad_x_slw = idx_m_slw * grad_x

        # gradient of tw
        idx_s_tw = (x_n <= t_n).float()
        idx_l_tw = (x_n >= t_p).float()
        idx_m_tw = torch.ones(size=idx_s_tw.shape, device=idx_s_tw.device) - idx_s_tw - idx_l_tw
        
        grad_s_tw = (idx_s_tw * t_n + idx_l_tw * t_p + idx_m_tw * (-x_n + x_q)) * grad_x / gs_tw
        grad_x_tw = idx_m_tw * grad_x

        # combination of both tw and slw
        grad_x = torch.where(mask == 1., grad_x_tw, grad_x_slw)
        grad_s = torch.where(mask == 1., grad_s_tw, grad_s_slw).sum().unsqueeze(dim=0)
        return grad_x, grad_s, None, None, None, None, None


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

    def forward(self, x):
        if self.training and self.do_init == 0:
            self.init_step(x)

        return Inferer.apply(x, self.step_abs, self.mask, self.q_n, self.q_p, self.t_n, self.t_p)

    def extra_repr(self):
        return 'nbits={}, q_n={}, q_p={}, t_n={}, t_p={}'.format(int(self.nbits[0]), int(self.q_n), int(self.q_p), int(self.t_n), int(self.t_p))
