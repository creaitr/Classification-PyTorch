
import sys
import time
import csv
import shutil
import pathlib
from copy import deepcopy
from os import remove
from os.path import isfile
from collections import OrderedDict

import torch


class AverageMeter(object):
    r"""Computes and stores the average and current value
    """
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        sys.stdout.write('\r')
        print(' '.join(entries), end='')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return 'Batch[' + fmt + '/' + fmt.format(num_batches) + '] '


def accuracy(output, target, topk=(1,)):
    r"""Computes the accuracy over the $k$ top predictions for the specified values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        ''' slower topk
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        '''
        # faster topk
        _, idx = output.sort(descending=True)
        pred = idx[:,:maxk]
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class Timer:
    def __init__(self):
        self.times = {}
    
    def start(self, *names):
        for name in names:
            self.times[str(name)] = time.time()

    def end(self, name: str):
        assert name in self.times.keys(), "There is no recorded time [key:%s]" % name
        self.times[name] = time.time() - self.times[name]

    def get(self, name: str) -> str:
        assert name in self.times.keys(), "There is no recorded time [key:%s]" % name
        temp = self.times[name]
        return '{}h {}m {:.2f}s'.format(
            int(temp // 3600), int((temp % 3600) // 60), temp % 60)
