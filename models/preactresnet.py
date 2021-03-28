'''Pre-activation ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(inplanes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return qnn.QuantConv2d(inplanes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation,
                     nbits=qcfg['bitw'], symmetric=qcfg['symmetric'],
                     centralize=qcfg['centralize'], do_mask=qcfg['do_mask'])


def conv1x1(inplanes, out_planes, stride=1):
    """1x1 convolution"""
    return qnn.QuantConv2d(inplanes, out_planes, kernel_size=1, stride=stride, bias=False,
                           nbits=qcfg['bitw'], symmetric=qcfg['symmetric'],
                           centralize=qcfg['centralize'], do_mask=qcfg['do_mask'])


def relu(inplace=False):
    """ReLU activation"""
    return qnn.QuantReLU(inplace=False, nbits=qcfg['bita'])


def bn(num_features):
    """Batch normalization 2D"""
    return nn.BatchNorm2d(num_features, momentum=0.1)


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, first_block=False):
        super(PreActBlock, self).__init__()
        self.first_block = first_block
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.bn1 = bn(inplanes)
        self.relu1 = relu(inplace=False)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = bn(planes)
        self.relu2 = relu(inplace=False)
        self.conv2 = conv3x3(planes, planes, stride=1)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.relu1(self.bn1(x))

        if self.downsample is not None:
            identity = self.downsample(out)
        elif self.first_block:
            identity = out
        else:
            identity = x
        
        out = self.conv1(out)
        out = self.relu2(self.bn2(out))
        out = self.conv2(out)
        out += identity
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, first_block=False):
        super(PreActBottleneck, self).__init__()
        self.first_block = first_block
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.bn1 = bn(inplanes)
        self.relu1 = relu(inplace=False)
        self.conv1 = conv1x1(inplanes, width)
        self.bn2 = bn(planes)
        self.relu2 = relu(inplace=False)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn3 = bn(planes)
        self.relu3 = relu(inplace=False)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.relu1(self.bn1(x))
        
        if self.downsample is not None:
            identity = self.downsample(out)
        elif self.first_block:
            identity = out
        else:
            identity = x
        
        out = self.conv1(out)
        out = self.relu2(self.bn2(out))
        out = self.conv2(out)
        out = self.relu3(self.bn3(out))
        out = self.conv3(out)
        out += identity
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, layers, image_size=224, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None):
        super(PreActResNet, self).__init__()
        self.block_name = str(block.__name__)

        self.inplanes = 64
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
        self.image_size = image_size
        
        if image_size == 32:
            self.conv1 = qnn.QuantConv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                                         bias=False, nbits=qcfg['first_conv_bitw'], symmetric=qcfg['symmetric'])
        elif image_size == 224:
            self.conv1 = qnn.QuantConv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                                         bias=False, nbits=qcfg['first_conv_bitw'], symmetric=qcfg['symmetric'])
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, first_block=True)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.bn = bn(512 * block.expansion)
        self.relu = relu(inplace=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = qnn.QuantLinear(512 * block.expansion, num_classes,
                                  nbits=qcfg['last_fc_bitw'], symmetric=qcfg['symmetric'])

        for m in self.modules():
            if isinstance(m, qnn.QuantConv2d):
                #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5), mode='fan_in', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, first_block=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
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
        if self.image_size == 224:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.relu(self.bn(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
    def forward(self, x):
        return self._forward_impl(x)


# Model configurations
model_cfgs = {
    18:  (PreActBlock, [2, 2, 2, 2]),
    34:  (PreActBlock, [3, 4, 6, 3]),
    50:  (PreActBottleneck, [3, 4, 6, 3]),
    101: (PreActBottleneck, [3, 4, 23, 3]),
    152: (PreActBottleneck, [3, 8, 36, 3]),
}


def preactresnet(dataset='cifar10', qnn=None, cfg=None):
    r"""
    Args:
        data (str): the name of datasets
    """
    # set quantization configurations
    globals()['qnn'] = qnn
    global qcfg
    qcfg = dict()
    qcfg['bitw'] = cfg.bitw
    qcfg['bita'] = cfg.bita
    qcfg['first_conv_bitw'] = cfg.first_conv_bitw
    qcfg['last_fc_bitw'] = cfg.last_fc_bitw
    qcfg['symmetric'] = cfg.symmetric
    qcfg['centralize'] = cfg.centralize
    qcfg['do_mask'] = cfg.mask_init or cfg.mask_update if cfg.centralize else False

    # set model configurations
    assert cfg.layers in model_cfgs.keys()
    block, layers = model_cfgs[cfg.layers]
    if cfg.dataset in ['cifar10', 'cifar100']:
        image_size = 32
        num_classes = int(cfg.dataset[5:])
    elif cfg.dataset in ['imagenet']:
        image_size = 224
        num_classes = 1000
    else:
        raise Exception('Undefined dataset for PreActResNet architecture.')

    model = PreActResNet(block, layers, image_size, num_classes)
    return model, image_size
