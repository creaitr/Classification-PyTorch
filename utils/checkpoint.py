import os
import shutil
import pathlib
from pathlib import Path
from copy import deepcopy
from itertools import chain
from collections import defaultdict

import torch


last_ckpt = 'ckpt.pth'
best_ckpt = 'best.pth'


def load_model(ckpt_path, model,
               state=None, optimizer=None, scheduler=None,
               device='cpu', strict=True):
    # load a checkpoint file
    ckpt = torch.load(ckpt_path, map_location=device)
    # load the model
    try:
        model.load_state_dict(ckpt['model'], strict=strict)
    except:
        model.module.load_state_dict(ckpt['model'], strict=strict)
    # load elses
    if state is not None:
        state.update(ckpt['state'])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(ckpt['scheduler'])


def load_optim_state(ckpt_path, model, optimizer, device='cpu'):
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


def load(cfg, model,
         state, optimizer, scheduler,
         logger, device='cpu'):
    r"""Load model for each situation: initialization, resume the training, and test.
    """
    if cfg.eval is not None:  # test the trained model
        assert cfg.eval.exists(), 'There is no directory for test:%s' % str(cfg.eval)
        assert (cfg.eval / best_ckpt).exists(), 'There is no checkpoint file:%s' % str(cfg.eval / best_ckpt)
        logger.print('Test the best trained checkpoint from %s' % str(cfg.eval))
        load_model(ckpt_path=cfg.eval / best_ckpt, model=model,
                   state=None, optimizer=None, scheduler=None,
                   device=device, strict=False)

    elif cfg.resume is not None:  # resume the train
        assert cfg.resume.exists(), 'There is no directory for resume:%s' % str(cfg.resume)
        assert (cfg.resume / last_ckpt).exists(), 'There is no checkpoint file:%s' % str(cfg.resume / last_ckpt)
        logger.print('Resume the training from %s' % str(cfg.resume))
        load_model(ckpt_path=cfg.resume / last_ckpt, model=model,
                   state=state, optimizer=optimizer, scheduler=scheduler,
                   device=device, strict=True)

    elif cfg.init is not None:    # load a pre-trained model
        assert cfg.init.exists(), 'There is no checkpoint file:%s' % str(cfg.init)
        logger.print('Initialize the model from %s' % str(cfg.init))
        load_model(ckpt_path=cfg.init, model=model,
                   state=None, optimizer=None, scheduler=None,
                   device=device, strict=False)
        if cfg.init_opt:
            load_optim_state(ckpt_path=cfg.init, model=model, optimizer=optimizer, device=device)


def save_last(log_path, model, state, optimizer, scheduler):
    new_state = {
        'state': state.copy(),
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    torch.save(new_state, log_path / last_ckpt)


def copy_last(log_path):
    shutil.copyfile(os.path.join(log_path, last_ckpt),
                    os.path.join(log_path, best_ckpt))


def save(acc1_val, logger, model, state, optimizer, scheduler):
    # save the last model
    save_last(logger.log_path, model,
              state, optimizer, scheduler)
    
    # save the best model
    is_best = acc1_val >= state['best_acc']
    state['best_acc'] = max(acc1_val, state['best_acc'])
    if is_best:
        copy_last(logger.log_path)
        logger.print('Best checkpoint is saved ...')
