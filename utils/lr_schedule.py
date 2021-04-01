import os
import math

import numpy as np

import torch


def set_scheduler(optimizer, cfg):
    r"""Sets the learning rate scheduler
    """
    if cfg.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.step_size, cfg.gamma)
    elif cfg.scheduler == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, cfg.milestones, cfg.gamma)
    elif cfg.scheduler == 'exp':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, cfg.gamma)
    elif cfg.scheduler == 'cosine':
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cfg.epochs)
        scheduler = WarmupCosineAnnealing(optimizer, epochs=cfg.epochs, warmup_epoch=0)
    elif cfg.scheduler == 'warmup_cosine':
        scheduler = WarmupCosineAnnealing(optimizer, epochs=cfg.epochs, warmup_epoch=5)
    else:
        raise ValueError('==> unavailable scheduler:%s' % cfg.scheduler)

    return scheduler


class WarmupCosineAnnealing(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, epochs, warmup_epoch=5, last_epoch=-1):
        if epochs <= 0 or not isinstance(epochs, int):
            raise ValueError("Expected positive integer epochs, but got {}".format(epochs))
        if warmup_epoch < 0 or not isinstance(warmup_epoch, int):
            raise ValueError("Expected positive integer or zero warmup_epoch, but got {}".format(warmup_epoch))
        self.epochs = epochs
        self.warmup_epoch = warmup_epoch

        super(WarmupCosineAnnealing, self).__init__(optimizer, last_epoch)

    def get_lr(self, epoch):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        
        lrs = []
        for base_lr in self.base_lrs:
            if epoch < self.warmup_epoch:
                lr = base_lr * (epoch + 1) / self.warmup_epoch
            else:
                lr = base_lr * (1 + math.cos(math.pi * (epoch - self.warmup_epoch) / (self.epochs - self.warmup_epoch))) / 2
            lrs.append(lr)
        return lrs

    def step(self, epoch=None):
        """Step could be called after every epoch or batch update
        Example:
            >>> scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
            >>> iters = len(dataloader)
            >>> for epoch in range(20):
            >>>     for i, sample in enumerate(dataloader):
            >>>         inputs, labels = sample['inputs'], sample['labels']
            >>>         optimizer.zero_grad()
            >>>         outputs = net(inputs)
            >>>         loss = criterion(outputs, labels)
            >>>         loss.backward()
            >>>         optimizer.step()
            >>>         scheduler.step(epoch + i / iters)
        """
        if epoch is None:
            epoch = self.last_epoch + 1
        elif epoch < 0:
            raise ValueError("Expected non-negative epoch, but got {}".format(epoch))
        self.last_epoch = math.floor(epoch)

        class _enable_get_lr_call:
            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        with _enable_get_lr_call(self):
            for i, data in enumerate(zip(self.optimizer.param_groups, self.get_lr(epoch))):
                param_group, lr = data
                param_group['lr'] = lr

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
