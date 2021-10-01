import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from .measure import Timer, AverageMeter, ProgressMeter, accuracy


class Trainer:
    def __init__(self, cfg, model, criterion, optimizer, lr_scheduler, loaders, logger):
        self.cfg = cfg
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loaders = loaders
        self.logger = logger

        self.timer = Timer()

        self.hooks = {
            'before_train': [],
            'before_epoch': [],
            'before_batch': [],
            'after_batch': [],
            'after_epoch': [],
            'after_train': [],
        }

        self.states = {} # e.g., best_acc
        self.reports = {} # e.g., epoch, acc1_trn
        self.memory = {} # e.g., iteration

    ### device ###
    def set_device(self):
        if self.cfg.device == 'gpu' and torch.cuda.is_available():
            torch.cuda.set_device(self.cfg.gpu[0])
            with torch.cuda.device(self.cfg.gpu[0]):
                self.model = self.model.cuda()
                self.criterion = self.criterion.cuda()
            self.model = nn.DataParallel(self.model, device_ids=self.cfg.gpu,
                                    output_device=self.cfg.gpu[0])
            cudnn.benchmark = True
            self.device = 'cuda:%d' % self.cfg.gpu[0]
        else:
            self.device = 'cpu'

    ### hook ###
    def register_hooks(self, loc, func, idx=-1):
        assert loc in self.hooks.keys(), f'The location of hook:{loc} should be defined correctly.'

        if not isinstance(func, list):
            func = [func]
        
        if idx == -1:
            for f in func:
                self.hooks[loc].append(f)
        else:
            for f in func[::-1]:
                self.hooks[loc].insert(idx, f)

    def run_hooks(self, loc):
        for func in self.hooks[loc]:
            func(self)
    
    ### run ###
    def train(self):
        # init parameters
        self.states['best_acc'] = 0
        self.memory['iterations'] = 0
        self.memory['batch_len_trn'] = len(self.loaders['train'])
        self.memory['batch_len_val'] = len(self.loaders['val'])

        # before train
        self.run_hooks('before_train')

        # start train
        self.logger.print(f'Training start ({self.cfg.arch}/{self.cfg.dataset})')
        for epoch in range(self.lr_scheduler.last_epoch, self.cfg.epochs):
            self.logger.print('')
            self.logger.print('* Epoch: {} (lr={})'.format(
                epoch, self.optimizer.param_groups[0]["lr"]))

            # init
            self.reports = {'epoch': epoch}

            # before epoch
            self.run_hooks('before_epoch')

            # train epoch
            self.train_epoch()
            # validate epoch
            self.validate_epoch()

            # after epoch
            self.run_hooks('after_epoch')

        # after train
        self.run_hooks('after_train')

        self.logger.print('Training end')
        return self.states['best_acc']

    def validate(self):
        # init parameters
        self.reports = {}
        self.memory['batch_len_val'] = len(self.loaders['val'])

        # before epoch
        self.run_hooks('before_epoch')

        # start validation
        self.logger.print(f'Validation start ({self.cfg.arch}/{self.cfg.dataset})')
        self.logger.print('')

        # validate epoch
        self.validate_epoch()

        # after epoch
        self.run_hooks('after_epoch')

        self.logger.print('Validation end')
        return self.reports['acc1_val']
    
    def train_epoch(self):
        r"""Train model each epoch
        """
        self.logger.print('[ Training ]')

        # init
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(self.memory['batch_len_trn'],
                                 batch_time, data_time,
                                 losses, top1, top5)

        # switch to train mode
        self.model.train()
        
        self.timer.start('epoch', 'data', 'batch')
        
        # main epoch
        for i, (input, target) in enumerate(self.loaders['train']):
            self.timer.end('data')
            data_time.update(self.timer.times['data'])

            # memorize
            self.memory['i'] = i
            self.memory['input'] = input
            self.memory['target'] = target

            # before batch
            self.run_hooks('before_batch')

            if self.cfg.device == 'gpu':
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            # model inference
            output = self.model(input)
            loss = self.criterion(output, target)

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            self.timer.end('batch')
            batch_time.update(self.timer.times['batch'])

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), n=input.size(0))
            top1.update(acc1[0], n=input.size(0))
            top5.update(acc5[0], n=input.size(0))

            # memorize
            self.reports['loss_trn'] = round(losses.avg, 14)
            self.reports['acc1_trn'] = round(top1.avg.item(), 4)
            self.reports['acc5_trn'] = round(top5.avg.item(), 4)
            self.memory['iterations'] += 1

            # after batch
            self.run_hooks('after_batch')

            if i % self.cfg.print_freq == 0:
                progress.print(i)

            self.timer.start('data', 'batch')

        self.timer.end('epoch')

        # report
        self.logger.print(f'''Results: Time {self.timer.get("epoch")}  Loss {self.reports['loss_trn']:.4e}  '''
                          f'''Acc@1 {self.reports['acc1_trn']:.3f}  Acc@5 {self.reports['acc5_trn']:.3f}''')

    def validate_epoch(self, batch_hooks=False):
        r"""Validate model each epoch and evaluation
        """
        self.logger.print('[ Validation ]')

        # init
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(self.memory['batch_len_val'],
                                 batch_time, data_time,
                                 losses, top1, top5)

        # switch to eval mode
        self.model.eval()
        
        self.timer.start('epoch', 'data', 'batch')
        
        # main epoch
        with torch.no_grad():
            for i, (input, target) in enumerate(self.loaders['val']):
                self.timer.end('data')
                data_time.update(self.timer.times['data'])

                if batch_hooks:
                    # memorize
                    self.memory['i'] = i
                    self.memory['input'] = input
                    self.memory['target'] = target

                    # before batch
                    self.run_hooks('before_batch')

                if self.cfg.device == 'gpu':
                    input = input.cuda(non_blocking=True)
                    target = target.cuda(non_blocking=True)

                # model inference
                output = self.model(input)
                loss = self.criterion(output, target)

                # measure elapsed time
                self.timer.end('batch')
                batch_time.update(self.timer.times['batch'])

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), n=input.size(0))
                top1.update(acc1[0], n=input.size(0))
                top5.update(acc5[0], n=input.size(0))

                # memorize
                self.reports['loss_val'] = round(losses.avg, 14)
                self.reports['acc1_val'] = round(top1.avg.item(), 4)
                self.reports['acc5_val'] = round(top5.avg.item(), 4)

                if batch_hooks:
                    # after batch
                    self.run_hooks('after_batch')

                if i % self.cfg.print_freq == 0:
                    progress.print(i)

                self.timer.start('data', 'batch')

        self.timer.end('epoch')

        # report
        self.logger.print(f'''Results: Time {self.timer.get("epoch")}  Loss {self.reports['loss_val']:.4e}  '''
                          f'''Acc@1 {self.reports['acc1_val']:.3f}  Acc@5 {self.reports['acc5_val']:.3f}''')

    def test(self):
        # init parameters
        self.memory['pred'] = []
        self.memory['batch_len_test'] = len(self.loaders['test'])

        # before epoch
        self.run_hooks('before_epoch')

        # start test
        self.logger.print(f'Test start ({self.cfg.arch}/{self.cfg.dataset})')
        self.logger.print('')

        # test epoch
        self.test_epoch()

        # after epoch
        self.run_hooks('after_epoch')

        logger.print('Test end')

    def test_epoch(self):
        r"""Test model
        """
        self.logger.print('[ Test ]')

        # init
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        progress = ProgressMeter(self.memory['batch_len_test'],
                                 batch_time, data_time)

        # switch to eval mode
        self.model.eval()
        
        self.timer.start('epoch', 'data', 'batch')

        # main epoch
        with torch.no_grad():
            for i, (input, target) in enumerate(self.loaders['val']):
                self.timer.end('data')
                data_time.update(self.timer.times['data'])

                if self.cfg.device == 'gpu':
                    input = input.cuda(non_blocking=True)
                    target = target.cuda(non_blocking=True)

                # model inference
                output = self.model(input)
                self.memory['pred'].append(output.detach().cpu().numpy())

                # measure elapsed time
                self.timer.end('batch')
                batch_time.update(self.timer.times['batch'])

                if i % self.cfg.print_freq == 0:
                    progress.print(i)

                self.timer.start('data', 'batch')

        self.timer.end('epoch')

        # report
        self.logger.print(f'''Results: Time {self.timer.get("epoch")}''')

    def analyze(self):
        # init parameters
        self.reports = {}
        self.memory['batch_len_val'] = len(self.loaders['val'])

        # before epoch
        self.run_hooks('before_epoch')

        # start analysis
        self.logger.print(f'Analysis start ({self.cfg.arch}/{self.cfg.dataset})')
        self.logger.print('')

        # analyze epoch
        self.validate_epoch(batch_hooks=True)

        # after epoch
        self.run_hooks('after_epoch')

        self.logger.print('Analysis end')
