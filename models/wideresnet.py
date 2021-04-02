import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.parameter import Parameter


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def relu(inplace=True):
    """ReLU activation"""
    return nn.ReLU(inplace=inplace)


def bn(num_features):
    """Batch normalization 2D"""
    return nn.BatchNorm2d(num_features)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1):
        super(BasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = bn(planes)
        self.relu1 = relu(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = bn(planes)
        self.relu2 = relu(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu2(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = bn(width)
        self.relu1 = relu(inplace=False)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = bn(width)
        self.relu2 = relu(inplace=False)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = bn(planes * self.expansion)
        self.relu3 = relu(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)

        return out


class WideResNet_Cifar(nn.Module):
    def __init__(self, block, layers, width_mult=1, num_classes=10, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None):
        super(WideResNet_Cifar, self).__init__()
        self.block_name = str(block.__name__)
        
        self.inplanes = 16
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3: 
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = bn(self.inplanes)
        self.relu1 = relu(inplace=False)

        self.layer1 = self._make_layer(block, 16 * width_mult, layers[0])
        self.layer2 = self._make_layer(block, 32 * width_mult, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 64 * width_mult, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion * width_mult, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                bn(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


# Model configurations
'''
model_cfgs = {
    18:  (BasicBlock, [2, 2, 2, 2]),
    34:  (BasicBlock, [3, 4, 6, 3]),
    50:  (Bottleneck, [3, 4, 6, 3]),
    101: (Bottleneck, [3, 4, 23, 3]),
    152: (Bottleneck, [3, 8, 36, 3]),
}
'''
model_cfgs_cifar = {
    16:  (BasicBlock, [2, 2, 2]),
    22:  (BasicBlock, [3, 3, 3]),
    28:  (BasicBlock, [4, 4, 4]),
    40:  (BasicBlock, [6, 6, 6]),
    52:  (BasicBlock, [8, 8, 8]),
}


def wideresnet(cfg):
    r"""
    Args:
        cfg: configuration
    """
    # set model configurations
    if data in ['cifar10', 'cifar100']:
        assert (cfg.layers - 4) % 6 == 0, "The number of layers should be 16, 22, 28, 40, 52, etc."
        assert cfg.width_mult == int(cfg.width_mult), "The width multiplier should be an integer value."
        n = int((cfg.layers - 4) / 6)
        layers = [n, n, n]
        image_size = 32
        num_classes = int(cfg.dataset[5:])
        model = WideResNet_Cifar(BasicBlock, layers, cfg.width_mult, num_classes)
        
    elif data == 'imagenet':
        model = None
        image_size = None
        raise Exception('Undefined dataset for WideResNet architecture.')

    else:
        raise Exception('Undefined dataset for WideResNet architecture.')
    

    return model, image_size