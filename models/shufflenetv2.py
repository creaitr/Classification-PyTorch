'''ShuffleNetV2 in PyTorch.

See the paper "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv2d(inplanes, out_planes, kernel_size=3, stride=1, padding=0, groups=1, bias=False):
    """convolution with padding"""
    return nn.Conv2d(inplanes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, groups=groups, bias=bias)


def relu(inplace=False):
    """ReLU activation"""
    return nn.ReLU(inplace=inplace)


class ConvBNReLU(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, act=True):
        super(ConvBNReLU, self).__init__()
        padding = (kernel_size - 1) // 2 if kernel_size > 1 else 0
        
        self.conv = conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups)
        self.bn = nn.BatchNorm2d(out_planes)
        self.act = relu(inplace=True) if act else None

    def forward(self, x):
        out = self.bn(self.conv(x))
        out = self.act(out) if self.act is not None else out
        return out


class ShuffleBlock(nn.Module):
    def __init__(self, groups=2):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N, C, H, W = x.size()
        g = self.groups
        return x.view(N, g, C//g, H, W).permute(0, 2, 1, 3, 4).reshape(N, C, H, W)


class SplitBlock(nn.Module):
    def __init__(self, ratio):
        super(SplitBlock, self).__init__()
        self.ratio = ratio

    def forward(self, x):
        c = int(x.size(1) * self.ratio)
        return x[:, :c, :, :], x[:, c:, :, :]


class BasicBlock(nn.Module):
    def __init__(self, in_channels, split_ratio=0.5):
        super(BasicBlock, self).__init__()
        self.split = SplitBlock(split_ratio)
        in_channels = int(in_channels * split_ratio)

        self.conv = nn.Sequential(
            # pw
            ConvBNReLU(in_channels, in_channels, kernel_size=1, stride=1, act=True),
            # dw
            ConvBNReLU(in_channels, in_channels, kernel_size=3, stride=1, groups=in_channels, act=False),
            # pw-linear
            ConvBNReLU(in_channels, in_channels, kernel_size=1, stride=1, act=True)
        )
        self.shuffle = ShuffleBlock()

    def forward(self, x):
        x1, x2 = self.split(x)
        out = torch.cat([x1, self.conv(x2)], 1)
        out = self.shuffle(out)
        return out


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(DownBlock, self).__init__()
        mid_channels = out_channels // 2
        
        # left
        self.conv1 = nn.Sequential(
            # dw
            ConvBNReLU(in_channels, in_channels, kernel_size=3, stride=stride, groups=in_channels, act=False),
            # pw-linear
            ConvBNReLU(in_channels, mid_channels, kernel_size=1, stride=1, act=True)
        )
        # right
        self.conv2 = nn.Sequential(
            # pw
            ConvBNReLU(in_channels, mid_channels, kernel_size=1, stride=1, act=True),
            # dw
            ConvBNReLU(mid_channels, mid_channels, kernel_size=3, stride=stride, groups=mid_channels, act=False),
            # pw-linear
            ConvBNReLU(mid_channels, mid_channels, kernel_size=1, stride=1, act=True)
        )
        self.shuffle = ShuffleBlock()

    def forward(self, x):
        out = torch.cat([self.conv1(x), self.conv2(x)], 1)
        out = self.shuffle(out)
        return out


class ShuffleNetV2(nn.Module):
    def __init__(self, out_channels, num_blocks, num_classes=1000):
        super(ShuffleNetV2, self).__init__()
        self.in_channels = 24

        self.conv1 = ConvBNReLU(3, self.in_channels, kernel_size=3, stride=2, act=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(out_channels[0], num_blocks[0])
        self.layer2 = self._make_layer(out_channels[1], num_blocks[1])
        self.layer3 = self._make_layer(out_channels[2], num_blocks[2])
        self.conv2 = ConvBNReLU(out_channels[2], out_channels[3],
                                kernel_size=1, stride=1, act=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(out_channels[3], num_classes)

    def _make_layer(self, out_channels, num_blocks):
        layers = [DownBlock(self.in_channels, out_channels)]
        for i in range(num_blocks):
            layers.append(BasicBlock(out_channels))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.conv2(out)

        out = self.avgpool(out)
        #out = torch.flatten(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ShuffleNetV2_CIFAR(nn.Module):
    def __init__(self, out_channels, num_blocks, num_classes=10):
        super(ShuffleNetV2_CIFAR, self).__init__()
        self.in_channels = 24

        self.conv1 = ConvBNReLU(3, self.in_channels, kernel_size=3, stride=1, act=True)
        
        self.layer1 = self._make_layer(out_channels[0], num_blocks[0])
        self.layer2 = self._make_layer(out_channels[1], num_blocks[1])
        self.layer3 = self._make_layer(out_channels[2], num_blocks[2])
        self.conv2 = ConvBNReLU(out_channels[2], out_channels[3],
                                kernel_size=1, stride=1, act=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(out_channels[3], num_classes)

    def _make_layer(self, out_channels, num_blocks):
        if self.in_channels == 24:
            layers = [DownBlock(self.in_channels, out_channels, stride=1)]
        else:
            layers = [DownBlock(self.in_channels, out_channels)]
        for i in range(num_blocks):
            layers.append(BasicBlock(out_channels))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.conv2(out)

        out = self.avgpool(out)
        #out = torch.flatten(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


model_cfgs = {
    0.5: {'out_channels': (48, 96, 192, 1024),
          'num_blocks': (3, 7, 3)},
    1:   {'out_channels': (116, 232, 464, 1024),
          'num_blocks': (3, 7, 3)},
    1.5: {'out_channels': (176, 352, 704, 1024),
          'num_blocks': (3, 7, 3)},
    2:   {'out_channels': (224, 488, 976, 2048),
          'num_blocks': (3, 7, 3)}
}


def set_model(cfg):
    r"""
    Args:
        cfg: configuration
    """
    # set model configurations
    assert cfg.width_mult in model_cfgs.keys(), "The width multiplier for ShuffleNetV2 should be 0.5, 1, 1.5, or 2."
    
    if cfg.dataset in ['cifar10', 'cifar100']:
        out_channels = model_cfgs[cfg.width_mult]['out_channels']
        num_blocks = model_cfgs[cfg.width_mult]['num_blocks']
        image_size = 32
        num_classes = int(cfg.dataset[5:])
        model = ShuffleNetV2_CIFAR(out_channels, num_blocks, num_classes)

    elif cfg.dataset in ['imagenet']:
        out_channels = model_cfgs[cfg.width_mult]['out_channels']
        num_blocks = model_cfgs[cfg.width_mult]['num_blocks']
        image_size = 224
        num_classes = 1000
        model = ShuffleNetV2(out_channels, num_blocks, num_classes)

    else:
        raise Exception('Undefined dataset for ShuffleNetV2 architecture.')

    return model, image_size