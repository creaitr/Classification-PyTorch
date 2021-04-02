import os 
import copy


def set_arch_name(cfg):
    r"""Set architecture name
    """
    arch_name = copy.deepcopy(cfg.arch)
    if cfg.arch in ['resnet', 'preactresnet']:
        arch_name += str(cfg.layers)
    elif cfg.arch in ['wideresnet']:
        arch_name += str(cfg.layers) + '-' + str(int(cfg.width_mult))
    elif cfg.arch in ['shufflenetv2', 'mobilenetv2', 'rexnet']:
        if cfg.width_mult != 1.0:
            arch_name += f'x{cfg.width_mult}'

    return arch_name