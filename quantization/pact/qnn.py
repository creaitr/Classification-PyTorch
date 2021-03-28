import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class Quantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, b):
        n = 2 ** b - 1
        x_q = (x * n).round() / n
        return x_q

    @staticmethod
    def backward(ctx, grad_x):
        return grad_x, None


class QuantConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
                 nbits=32, **kwargs):
        super(QuantConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                     padding, dilation, groups, bias, padding_mode)
        
        assert nbits > 0 and nbits < 33
        self.nbits = Parameter(torch.Tensor(1), requires_grad=False)
        self.nbits.fill_(nbits)
    
    def forward(self, input):
        if self.nbits == 32:
            quantized_weight = self.weight
        else:
            w_t = torch.tanh(self.weight)
            w_t = w_t / w_t.abs().max() * 0.5 + 0.5

            quantized_weight = 2. * Quantizer.apply(w_t, self.nbits) - 1.
        return self._conv_forward(input, quantized_weight)

    def extra_repr(self):
        s_prefix = super(QuantConv2d, self).extra_repr()
        return '{}, nbits={}'.format(s_prefix, int(self.nbits[0]))


class QuantLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, nbits=32, **kwargs):
        super(QuantLinear, self).__init__(in_features, out_features, bias)

        assert nbits > 0 and nbits < 33
        self.nbits = Parameter(torch.Tensor(1), requires_grad=False)
        self.nbits.fill_(nbits)

    def forward(self, input):
        if self.nbits == 32:
            quantized_weight = self.weight
            bias = self.bias
        else:
            w_t = torch.tanh(self.weight)
            w_t = w_t / w_t.abs().max() * 0.5 + 0.5

            quantized_weight = 2. * Quantizer.apply(w_t, self.nbits) - 1.
            bias = self.bias
        return F.linear(input, quantized_weight, bias)

    def extra_repr(self):
        s_prefix = super(QuantLinear, self).extra_repr()
        return '{}, nbits={}'.format(s_prefix, int(self.nbits[0]))


class QuantReLU(nn.ReLU):
    def __init__(self, inplace=False, nbits=32, **kwargs):
        super(QuantReLU, self).__init__(inplace)

        assert nbits > 0 and nbits < 33
        self.nbits = Parameter(torch.Tensor(1), requires_grad=False)
        self.nbits.fill_(nbits)

        if nbits != 32:
            self.alpha = Parameter(torch.Tensor(1))
            self.alpha.fill_(10.)

    def forward(self, input):
        if self.nbits == 32:
            return super(QuantReLU, self).forward (input)
        else:
            x_c = torch.clamp(input, 0., self.alpha)

            output = Quantizer.apply(x_c / self.alpha, self.nbits) * self.alpha
            return output

    def extra_repr(self):
        s_prefix = super(QuantReLU, self).extra_repr()
        return '{}, nbits={}'.format(s_prefix, int(self.nbits[0]))
