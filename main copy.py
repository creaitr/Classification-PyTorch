# system
import os
from pathlib import Path
import datetime
import copy
# file I/O
import yaml
import json
# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
#from torchsummary import summary
# parsing
from config import parse_arguments
# r/w training
import models
import quantization
import dataset
# utils
from utils import *

# init global parameters
timer = Timer()


def main():
    ####    Argument parsing    ####
    cfg = parse_arguments()
    
    # set the name of model
    arch_name = models.utils.set_arch_name(cfg)
    
    ####    Init logger    ####
    if cfg.resume is None and cfg.eval is None:
        log_path = Path('logs') / arch_name / cfg.dataset
        if cfg.name is not None:
            log_path = log_path / cfg.name
            if cfg.idx is not None:
                log_path = log_path / str(cfg.idx)
        assert not log_path.exists(), 'PATH:%s is already exist.' % (log_path)
    else:
        log_path = Path(cfg.resume) if cfg.eval is None else Path(cfg.eval)
        assert log_path.exists(), 'There is no PATH:%s.' % (log_path)
    # make a logger
    logger = Logger(log_path, cfg)
    
    ####    Construct a model    ####
    # set a quantizer
    logger.print('Setting quantization mode to %s ...' % cfg.quantization)
    quantizer = quantization.__dict__[cfg.quantization]
    # set a model
    logger.print('Building a model (%s) ...' % arch_name)
    model, image_size = models.__dict__[cfg.arch](cfg.dataset, quantizer.qnn, cfg)
    #summary(copy.deepcopy(model), (3, image_size, image_size), device='cpu')   # since an inference while summarize, the initial state couldn't be properly updated.
    # set a loss
    criterion = nn.CrossEntropyLoss()

    ####    Set device    ####
    if len(cfg.gpu) != 0 and torch.cuda.is_available():
        torch.cuda.set_device(cfg.gpu[0])
        with torch.cuda.device(cfg.gpu[0]):
            model = model.cuda()
            criterion = criterion.cuda()
        model = nn.DataParallel(model, device_ids=cfg.gpu,
                                output_device=cfg.gpu[0])
        cudnn.benchmark = True
        device = 'cuda:%d' % cfg.gpu[0]
    else:
        device = 'cpu'
    
    ####    Set optimizer and lr scheduler    ####
    optimizer = set_optimizer(model, cfg)
    scheduler = set_scheduler(optimizer, cfg)
    
    ####    Load a dataset    ####
    logger.print('Preparing dataset %s ...' % cfg.dataset)
    train_loader, val_loader = dataset.get_dataset(cfg, image_size)
    
    ####    Load a checkpoint    ####
    state = {
        'best_acc': 0,}
    load(cfg=cfg, model=model,
         state=state, optimizer=optimizer, scheduler=scheduler,
         logger=logger, device=device)

    ####    Initialize quantizers    ####
    if cfg.quant_init and 'initialize' in quantizer.__dict__.keys() and cfg.run_type == 'train':
        quantizer.initialize(model)
    if cfg.centralize and cfg.mask_init:
        #quantizer.init_mask(cfg, model)
        quantizer.update_mask(cfg, model, cfg.slw_rate)

    ####    Run the main    ####
    if cfg.run_type == 'train':
        # before train
        logger.print('Training start: {}/{}'.format(arch_name, cfg.dataset))

        # init parameters
        global iterations
        iterations = 0

        for epoch in range(scheduler.last_epoch, cfg.epochs):
            # before epoch
            logger.print('')
            logger.print('* Epoch: {} (lr={})'.format(
                epoch, optimizer.param_groups[0]["lr"]))
            
            # training
            logger.print('[ Training ]')
            loss_trn, acc1_trn, acc5_trn = train(
                cfg, train_loader,
                epoch=epoch, model=model, criterion=criterion,
                optimizer=optimizer, scheduler=scheduler,
                logger=logger, device=device,
                quantizer=quantizer)
            
            # validation
            logger.print('[ Validation ]')
            loss_val, acc1_val, acc5_val = validate(
                cfg, val_loader,
                epoch=epoch, model=model, criterion=criterion,
                logger=logger, device=device)

            # after epoch
            # save the last model
            save_last(logger.log_path, model,
                      state, optimizer, scheduler)
            # save the best model
            is_best = acc1_val >= state['best_acc']
            state['best_acc'] = max(acc1_val, state['best_acc'])
            if is_best:
                copy_last(logger.log_path)
                logger.print('Best checkpoint is saved ...')

            # summary
            logger.summarize({'epoch': epoch,
                'loss_trn': loss_trn, 'acc1_trn': acc1_trn, 'acc5_trn': acc5_trn,
                'loss_val': loss_val, 'acc1_val': acc1_val, 'acc5_val': acc5_val})

        # after train
        logger.print('Training end')
        return state['best_acc']

    elif cfg.run_type == 'evaluate':
        # before evaluate
        logger.print('Evaluate start: {}/{}\n'.format(arch_name, cfg.dataset))

        # validation
        logger.print('')
        logger.print('[ Evaluate ]')
        loss_val, acc1_val, acc5_val = validate(
            cfg, val_loader,
            epoch=0, model=model, criterion=criterion,
            logger=logger, device=device)

        # summary
        logger.summarize({'epoch': 0,
            'loss_val': loss_val, 'acc1_val': acc1_val, 'acc5_val': acc5_val})

        # after evaluate
        logger.print('Evaluate end')
        return acc1_val

    elif cfg.run_type == 'extract':
        feature_path = Path('features') / str(logger.log_path).replace('/', '_')
        if not feature_path.exists():
            feature_path.mkdir()
        
        def save_feature(self, input, output):
            mat = output.cpu().numpy()
            if mat.ndim == 4:
                mat = np.mean(mat, axis=(2,3))
            
            if self.extracted is None:
                self.extracted = mat
            else:
                self.extracted = np.vstack((self.extracted, mat))

        for name, module in model.named_modules():
            if isinstance(module, quantizer.qnn.QuantConv2d) or isinstance(module, quantizer.qnn.QuantLinear):
                module.extracted = None
                module.register_forward_hook(save_feature)
                
                temp_path = feature_path / name
                if not temp_path.exists():
                    temp_path.mkdir()
        
        logger.print('Extract start')

        k = 0
        model.eval()
        with torch.no_grad():
            for i, (input, target) in enumerate(val_loader):
                if device is not 'cpu':
                    input = input.cuda(non_blocking=True)
                    target = target.cuda(non_blocking=True)

                output = model(input)
        
                if model.module.conv1.extracted.shape[0] > 10000:
                    for name, module in model.named_modules():
                        if isinstance(module, quantizer.qnn.QuantConv2d) or isinstance(module, quantizer.qnn.QuantLinear):
                            np.save(feature_path / name / (str(k).zfill(1) + '.npy'), module.extracted)
                            module.extracted = None
                    k += 1
            k += 1
            for name, module in model.named_modules():
                if isinstance(module, quantizer.qnn.QuantConv2d) or isinstance(module, quantizer.qnn.QuantLinear):
                    np.save(feature_path / name / (str(k).zfill(1) + '.npy'), module.extracted)
        
        logger.print('Extract end')


####    Train    ####
def train(cfg, train_loader, epoch, model, criterion, optimizer, scheduler, logger, device='cpu', quantizer=None, **kwargs):
    r"""Train model each epoch
    """
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time,
                             losses, top1, top5)

    # switch to train mode
    model.train()

    # before epoch
    iters = len(train_loader)
    timer.start('train', 'data', 'batch', 'print')

    # main epoch
    for i, (input, target) in enumerate(train_loader):
        timer.end('data')
        data_time.update(timer.times['data'])

        if device is not 'cpu':
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        # update mask
        if cfg.centralize and cfg.mask_update:
            if globals()['iterations'] % cfg.cent_freq == 0 and epoch < 0.7 * cfg.epochs:
                quantizer.update_mask(cfg, model, cfg.slw_rate)
                #quantizer.update_mask_step(model)

        output = model(input)
        loss = criterion(output, target)
        if cfg.cent_reg and 'regularize' in quantizer.__dict__.keys():
            loss += quantizer.regularize_th(cfg, epoch + i / iters, model)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), n=input.size(0))
        top1.update(acc1[0], n=input.size(0))
        top5.update(acc5[0], n=input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if cfg.sched_batch:
            scheduler.step(epoch + i / iters)

        # measure elapsed time
        timer.end('batch')
        batch_time.update(timer.times['batch'])

        # after batch
        globals()['iterations'] += 1
        if (time.time() - timer.times['print']) % 60 > 1 or i == len(train_loader) - 1:   # over 1s
            progress.print(i)
            timer.start('print')

        timer.start('data', 'batch')

    # after epoch
    if not cfg.sched_batch:
        scheduler.step()
    timer.end('train')

    # report
    print()
    logger.print('Results: Time {}  Loss {:.4e}  Acc@1 {:.3f}  Acc@5 {:.3f}'.format(
        timer.get('train'), losses.avg, top1.avg, top5.avg))
    # print center probability
    if cfg.centralize:
        if cfg.cent_reg:
            cent_prob = quantizer.cal_cent_prob_step(model)
        elif cfg.mask_init:
            cent_prob = quantizer.cal_cent_prob_static_mask(model)
        elif cfg.mask_update:
            cent_prob = quantizer.cal_cent_prob_dynamic_mask(model)
        logger.print('center prob: {:.4f}%'.format(cent_prob * 100))
    return losses.avg, round(top1.avg.item(), 4), round(top5.avg.item(), 4)


####    Validate    ####
def validate(cfg, val_loader, epoch, model, criterion, logger, device='cpu', **kwargs):
    r"""Validate model each epoch and evaluation
    """
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, data_time,
                             losses, top1, top5)

    # switch to train mode
    model.eval()

    # before validate
    timer.start('validate', 'data', 'batch', 'print')

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            timer.end('data')
            data_time.update(timer.times['data'])

            if device is not 'cpu':
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), n=input.size(0))
            top1.update(acc1[0], n=input.size(0))
            top5.update(acc5[0], n=input.size(0))

            # measure elapsed time
            timer.end('batch')
            batch_time.update(timer.times['batch'])

            # after batch
            if (time.time() - timer.times['print']) % 60 > 1 or i == len(val_loader) - 1:   # over 1s
                progress.print(i)
                timer.start('print')

            timer.start('data', 'batch')
    
    # after validate
    timer.end('validate')

    # report
    print()
    logger.print('Results: Time {}  Loss {:.4e}  Acc@1 {:.3f}  Acc@5 {:.3f}'.format(
        timer.get('validate'), losses.avg, top1.avg, top5.avg))
    
    return losses.avg, round(top1.avg.item(), 4), round(top5.avg.item(), 4)


if __name__ == '__main__':
    result = main()
    if result is not None:
        print('Best Acc1: {:.3f}%'.format(result))
