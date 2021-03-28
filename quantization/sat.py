import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn.init as init


class WeightQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, b):
        n = 2 ** b - 1
        x_q = (x * n).round() / n
        return x_q

    @staticmethod
    def backward(ctx, grad_x):
        return grad_x, None


class ActivationQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, b, a):
        x_c = torch.clamp(input, 0., a) / a

        n = 2 ** b - 1
        x_q = (x_c * n).round() / n

        ctx.save_for_backward(x_c, x_q, a)
        return x_q * a

    @staticmethod
    def backward(ctx, grad_x):
        x_c, x_q, a = ctx.saved_tensors
        
        idx_s = (x_c <= 0.).float()
        idx_l = (x_c >= 1.).float()
        idx_m = torch.ones(size=idx_s.shape, device=idx_s.device) - idx_s - idx_l
        
        grad_x = idx_m * grad_x
        grad_s = ((idx_l + idx_m * (x_q - x_c)) * grad_x).sum(dim=0)
        return grad_x, grad_s, None, None


class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
                 nbits=32):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                     padding, dilation, groups, bias, padding_mode)
        
        assert nbits > 0 and nbits < 33
        self.nbits = nbits
    
    def forward(self, input):
        if self.nbits == 32:
            weight = self.weight
        else:
            w_t = torch.tanh(self.weight)
            w_t = w_t / w_t.abs().max() * 0.5 + 0.5

            weight = 2. * WeightQuantizer.apply(w_t, self.nbits) - 1.
            with torch.no_grad:
                weight = weight / (weight.numel() * weight.square().mean()).sqrt()
        return super(Conv2d, self)._conv_forward(input, weight)

    def extra_repr(self):
        s_prefix = super(Conv2d, self).extra_repr()
        return '{}, nbits={}'.format(s_prefix, self.nbits)


class ReLU(nn.ReLU):
    def __init__(self, inplace=False, nbits=32):
        super(ReLU, self).__init__(inplace)

        assert nbits > 0 and nbits < 33
        self.nbits = nbits
        if nbits != 32:
            self.alpha = Parameter(torch.Tensor(1))
            init.constant(self.alpha, 10.)
        else:
            self.register_parameter('alpha', None)

    def forward(self, input):
        if self.nbits == 32:
            return super(ReLU, self).forward (input)
        else:
            output = ActivationQuantizer.apply(input, self.nbits, self.alpha)
            return output

    def extra_repr(self):
        s_prefix = super(ReLU, self).extra_repr()
        return '{}, nbits={}'.format(s_prefix, self.nbits)


class Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, nbits=32):
        super(Linear, self).__init__(in_features, out_features, bias)

        assert nbits > 0 and nbits < 33
        self.nbits = nbits

    def forward(self, input):
        if self.nbits == 32:
            weight = self.weight
            bias = self.bias
        else:
            w_t = torch.tanh(self.weight)
            w_t = w_t / w_t.abs().max() * 0.5 + 0.5

            weight = 2. * WeightQuantizer.apply(w_t, self.nbits) - 1.
            with torch.no_grad:
                weight = weight / (weight.numel() * weight.square().mean()).sqrt()
            bias = self.bias
        return F.linear(input, weight, bias)

    def extra_repr(self):
        s_prefix = super(Linear, self).extra_repr()
        return '{}, nbits={}'.format(s_prefix, self.nbits)


class SAT:
    Conv2d = Conv2d
    Linear = Linear
    ReLU = ReLU
