import os
import shutil
import pathlib
from pathlib import Path
from copy import deepcopy
from itertools import chain
from collections import defaultdict

import torch

last_ckpt = 'last.pth'
best_ckpt = 'best.pth'


### load ###
def load_init(trainer):
    assert trainer.cfg.load.exists(), f'The cfg.load:{trainer.cfg.load} is not exists.'
    assert trainer.cfg.load.is_file(), f'The checkpoint:{trainer.cfg.load} should be a file, not directory.'

    trainer.logger.print(f'Initialize the model from the checkpoint:{trainer.cfg.load}')
    load_model(ckpt_path=trainer.cfg.load, model=trainer.model,
               states=None, optimizer=None, lr_scheduler=None,
               device=trainer.device, strict=False)


def load_resume(trainer):
    assert trainer.cfg.load.is_dir(), f'The cfg.load:{trainer.cfg.load} should be a directory to resume training.'
    ckpt_path = trainer.cfg.load / last_ckpt
    assert ckpt_path.exists(), f'There is no the last checkpoint file:{ckpt_path}'

    trainer.logger.print(f'Resume the training from the dir:{trainer.cfg.load}')
    load_model(ckpt_path=ckpt_path, model=trainer.model,
               states=trainer.states, optimizer=trainer.optimizer, lr_scheduler=trainer.lr_scheduler,
               device=trainer.device, strict=True)


def load_valid(trainer):
    ckpt_path = trainer.cfg.load / best_ckpt
    assert ckpt_path.exists(), f'There is no checkpoint file:{ckpt_path}'
        
    trainer.logger.print(f'Load the model with the checkpoint:{ckpt_path}')
    load_model(ckpt_path=ckpt_path, model=trainer.model,
               states=None, optimizer=None, lr_scheduler=None,
               device=trainer.device, strict=False)


def load_model(ckpt_path, model,
               states=None, optimizer=None, lr_scheduler=None,
               device='cpu', strict=True):
    # load a checkpoint file
    ckpt = torch.load(ckpt_path, map_location=device)
    # load the model
    try:
        model.load_state_dict(ckpt['model'], strict=strict)
    except:
        model.module.load_state_dict(ckpt['model'], strict=strict)
    # load elses
    if states is not None:
        states.update(ckpt['states'])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer'])
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(ckpt['lr_scheduler'])


def load_optimizer_states(ckpt_path, model, optimizer, device='cpu'):
    '''
        usage example: load_optimizer_states(ckpt_path=cfg.init, model=model, optimizer=optimizer, device=device)
    '''
    # load a state_dict of the optimizer
    state_dict = torch.load(ckpt_path, map_location=device)['optimizer']
    # deepcopy, to be consistent with module API
    state_dict = deepcopy(state_dict)

    # Validate the state_dict
    groups = optimizer.param_groups
    saved_groups = state_dict['param_groups']

    if len(groups) != len(saved_groups):
        raise ValueError("loaded state dict has a different number of "
                         "parameter groups")

    valid_ids = []
    for name, item in model.named_parameters():
        if 'weight' in name or 'bias' in name:
            valid_ids.append(id(item))

    params = [[p for p in g['params'] if id(p) in valid_ids] for g in groups]
    saved_params = [g['params'] for g in saved_groups]

    param_lens = [len(g) for g in params]
    saved_lens = [len(g) for g in saved_params]
    if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
        raise ValueError("loaded state dict contains a parameter group "
                         "that doesn't match the size of optimizer's group")

    # Update the state
    id_map = {old_id: p for old_id, p in
              zip(chain(*saved_params), chain(*params))}

    def cast(param, value):
        r"""Make a deep copy of value, casting all tensors to device of param."""
        if isinstance(value, torch.Tensor):
            # Floating-point types are a bit special here. They are the only ones
            # that are assumed to always match the type of params.
            if param.is_floating_point():
                value = value.to(param.dtype)
            value = value.to(param.device)
            return value
        elif isinstance(value, dict):
            return {k: cast(param, v) for k, v in value.items()}
        elif isinstance(value, container_abcs.Iterable):
            return type(value)(cast(param, v) for v in value)
        else:
            return value

    # Copy state assigned to params (and cast tensors to appropriate types).
    # State that is not assigned to params is copied as is (needed for
    # backward compatibility).
    state = defaultdict(dict)
    for k, v in state_dict['state'].items():
        if k in id_map:
            param = id_map[k]
            state[param] = cast(param, v)
        else:
            state[k] = v

    optimizer.__setstate__({'state': state, 'param_groups': groups})


### save ###
def save_train(trainer):
    # update states
    is_best = trainer.reports['acc1_val'] >= trainer.states['best_acc']
    if is_best:
        trainer.states['best_acc'] = max(trainer.reports['acc1_val'],
                                         trainer.states['best_acc'])
    
    # save the last model
    save_last(trainer.logger.log_path, trainer.model,
              trainer.states, trainer.optimizer, trainer.lr_scheduler)
        
    # save the best model
    if is_best:
        trainer.logger.print('Best checkpoint is changed!')
        save_best(trainer.logger.log_path)


def save_last(log_path, model, states, optimizer, lr_scheduler):
    new_states = {
        'states': states.copy(),
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict()
    }
    torch.save(new_states, log_path / last_ckpt)


def save_best(log_path, source=None):
    if source == None:
        # copy the last checkpoint
        shutil.copyfile(str(log_path / last_ckpt), str(log_path / best_ckpt))
    else:
        # copy the source
        shutil.copyfile(str(source), str(log_path / best_ckpt))


def save_pred(trainer):
    raise NotImplementedError("This function is not implemented.")