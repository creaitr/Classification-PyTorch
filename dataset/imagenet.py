import torch

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageNet as IMAGENET

# for ignore ImageNet PIL EXIF UserWarning
import warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


class ImageNet(object):  
    def __init__(self, cfg):
        self.datapath = cfg.datapath

        self.image_size = None
        self.num_classes = 1000
        self.batch_size = cfg.batch_size
        self.num_workers = cfg.workers
        if len(cfg.gpu) != 0 and torch.cuda.is_available():
            self.cuda = True
        else:
            self.cuda = False

    def get_loader(self, image_size=224):
        self.image_size = image_size
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(self.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        transform_val = transforms.Compose([
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])

        trainset = IMAGENET(
            root=self.datapath, split='train', transform=transform_train)
        valset = IMAGENET(
            root=self.datapath, split='val', transform=transform_val)

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