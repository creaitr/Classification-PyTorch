import torch

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100


class Cifar100(object):  
    def __init__(self, cfg):
        self.datapath = cfg.datapath

        self.image_size = 32
        self.num_classes = 100
        self.batch_size = cfg.batch_size
        self.num_workers = cfg.workers
        if len(cfg.gpu) != 0 and torch.cuda.is_available():
            self.cuda = True
        else:
            self.cuda = False

    def get_loader(self):
        normalize = transforms.Normalize(
            mean=[0.5071, 0.4865, 0.4409],
            std=[0.2673, 0.2564, 0.2762])
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        trainset = CIFAR100(
            root=self.datapath, train=True, download=True,
            transform=transform_train)
        valset = CIFAR100(
            root=self.datapath, train=False, download=True,
            transform=transform_val)

        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True if self.cuda else False)
        val_loader = torch.utils.data.DataLoader(
            valset,
            batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if self.cuda else False)

        return train_loader, val_loader