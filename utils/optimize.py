import torch
import torch.optim as optim


def set_optimizer(model, cfg):
    r"""Sets the optimizer
    """
    if cfg.step_momentum is not None:
        return set_optimizer_step(model, cfg)
    
    if cfg.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=cfg.lr,
                              momentum=cfg.momentum, weight_decay=cfg.weight_decay,
                              nesterov=cfg.nesterov)
    elif cfg.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr,
                               betas=(cfg.momentum, 0.999),
                               weight_decay=cfg.weight_decay)
    return optimizer


def get_others(model):
    for name, param in model.named_parameters():
        if 'step' not in name:
            yield param


def get_steps(model):
    for name, param in model.named_parameters():
        if 'step' in name:
            yield param


def set_optimizer_step(model, cfg):
    r"""Sets the optimizer
    """
    if cfg.optimizer == 'SGD':
        optimizer = optim.SGD([
            {'params': get_others(model)},
            {'params': get_steps(model), 'momentum': cfg.step_momentum}],
            lr=cfg.lr, momentum=cfg.momentum,
            weight_decay=cfg.weight_decay, nesterov=cfg.nesterov)
    else:
        raise Exception('unkown optimizer.')
    
    return optimizer
