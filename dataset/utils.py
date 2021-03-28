from .cifar10 import Cifar10
from .cifar100 import Cifar100
from .imagenet import ImageNet


def get_dataset(cfg, image_size=224):
    r"""Dataloader for training/validation
    """
    if cfg.dataset == 'cifar10':
        return Cifar10(cfg).get_loader()
    elif cfg.dataset == 'cifar100':
        return Cifar100(cfg).get_loader()
    elif cfg.dataset == 'imagenet':
        return ImageNet(cfg).get_loader(image_size)