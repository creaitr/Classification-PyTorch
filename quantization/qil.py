import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn.init as init


class Quantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, b):
        n = 2 ** b - 1
        x_q = (x * n).round() / n
        return x_q

    @staticmethod
    def backward(ctx, grad_x):
        return grad_x, None


class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
                 nbits=32):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                     padding, dilation, groups, bias, padding_mode)
        
        assert nbits > 0 and nbits < 33
        self.nbits = nbits
        if nbits != 32:
            self.cw = Parameter(torch.Tensor(1))
            self.dw = Parameter(torch.Tensor(1))
            self.gamma = Parameter(torch.Tensor(1))
            init.constant(self.gamma, 1.)
            self.register_buffer('init_state', torch.zeros(1))
        else:
            self.register_parameter('cw', None)
            self.register_parameter('dw', None)
            self.register_parameter('gamma', None)

    def initialize_step(self):
        self.step.data.copy_(
            2. * self.weight.abs().mean() / math.sqrt(2 ** (self.nbits - 1) - 1)
        )
        self.init_state.fill_(1)
    
    def forward(self, input):
        if self.nbits == 32:
            weight = self.weight
        else:
            if self.training and self.init_state == 0:
                self.initialize_step()

            alpha = 0.5 / self.dw
            beta = -0.5 * self.cw / self.dw + 0.5
            w_t = torch.clamp(alpha * self.weight.abs() + beta, 0., 1.)
            w_t = w_t.pow(self.gamma)
            w_t = w_t * self.weight.sign()

            weight = Quantizer.apply(w_t, self.nbits - 1)
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
            self.cx = Parameter(torch.Tensor(1))
            self.dx = Parameter(torch.Tensor(1))
            self.register_buffer('init_state', torch.zeros(1))
        else:
            self.register_parameter('cx', None)
            self.register_parameter('dx', None)

    def initialize_step(self, input):
        self.step.data.copy_(
            2. * input.abs().mean() / math.sqrt(2 ** self.nbits - 1)
        )
        self.init_state.fill_(1)

    def forward(self, input):
        if self.nbits == 32:
            return super(ReLU, self).forward (input)
        else:
            if self.training and self.init_state == 0:
                self.initialize_step(input)

            alpha = 0.5 / self.dx
            beta = -0.5 * self.cx / self.dx + 0.5
            x_t = torch.clamp(alpha * input + beta, 0., 1.)

            output = Quantizer.apply(x_t, self.nbits)
            return output

    def extra_repr(self):
        s_prefix = super(ReLU, self).extra_repr()
        return '{}, nbits={}'.format(s_prefix, self.nbits)


class Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, nbits=32):
        super(Linear, self).__init__(in_features, out_features, bias)

        assert nbits > 0 and nbits < 33
        self.nbits = nbits
        if nbits != 32:
            self.cw = Parameter(torch.Tensor(1))
            self.dw = Parameter(torch.Tensor(1))
            self.gamma = Parameter(torch.Tensor(1))
            init.constant(self.gamma, 1.)
            self.register_buffer('init_state', torch.zeros(1))
        else:
            self.register_parameter('cw', None)
            self.register_parameter('dw', None)
            self.register_parameter('gamma', None)

    def initialize_step(self):
        self.step.data.copy_(
            2. * self.weight.abs().mean() / math.sqrt(2 ** (self.nbits - 1) - 1)
        )
        self.init_state.fill_(1)

    def forward(self, input):
        if self.nbits == 32:
            weight = self.weight
            bias = self.bias
        else:
            if self.training and self.init_state == 0:
                self.initialize_step()

            alpha = 0.5 / self.dw
            beta = -0.5 * self.cw / self.dw + 0.5
            w_t = torch.clamp(alpha * self.weight.abs() + beta, 0., 1.)
            w_t = w_t.pow(self.gamma)
            w_t = w_t * self.weight.sign()

            weight = Quantizer.apply(w_t, self.nbits - 1)
            bias = self.bias
        return F.linear(input, weight, bias)

    def extra_repr(self):
        s_prefix = super(Linear, self).extra_repr()
        return '{}, nbits={}'.format(s_prefix, self.nbits)


class QIL:
    Conv2d = Conv2d
    Linear = Linear
    ReLU = ReLU
