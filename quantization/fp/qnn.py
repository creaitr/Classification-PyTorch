import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class QuantConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', **kwargs):
        super(QuantConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                     padding, dilation, groups, bias, padding_mode)


class QuantLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, **kwargs):
        super(QuantLinear, self).__init__(in_features, out_features, bias)


class QuantReLU(nn.ReLU):
    def __init__(self, inplace=False, **kwargs):
        super(QuantReLU, self).__init__(inplace)


class QuantReLU6(nn.ReLU):
    def __init__(self, inplace=False, **kwargs):
        super(QuantReLU6, self).__init__(inplace)


class QuantIdentity(nn.Identity):
    def __init__(self, **kwargs):
        super(QuantIdentity, self).__init__()
