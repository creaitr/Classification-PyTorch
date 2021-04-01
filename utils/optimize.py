import torch
import torch.optim as optim


def set_optimizer(model, cfg):
    r"""Sets the optimizer
    """
    if cfg.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=cfg.lr,
                              momentum=cfg.momentum, weight_decay=cfg.weight_decay,
                              nesterov=cfg.nesterov)
    elif cfg.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr,
                               betas=(cfg.momentum, 0.999),
                               weight_decay=cfg.weight_decay)
    return optimizer
