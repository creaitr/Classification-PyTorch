import os
import time
import math

import numpy as np
import torch

from .qnn import *


def init_mask(cfg, model):
    for name, module in model.named_modules():
        if type(module) in [QuantConv2d, QuantLinear]:
            if module.centralize:
                x = module.weight
                s = module.quantizer.step_abs
                m = module.quantizer.mask
                
                # init mask
                m.data.copy_(
                   torch.where(x.abs() < s * 1.5, torch.ones_like(x), torch.zeros_like(x)))


def update_mask_step(model):
    state = model.state_dict()
    for name, item in model.named_parameters():
        if 'mask' in name:
            weight = state[name.replace('quantizer.mask', 'weight')]
            step = state[name.replace('mask', 'step')]
            
            new_mask = torch.where(weight.abs() < step.abs() * 1.5, torch.ones_like(item), torch.zeros_like(item))
            item.data.copy_(new_mask)


def update_mask(cfg, model, slw_rate=0.0):
    threshold = get_threshold(model, slw_rate)
    update(model, threshold)


def norm_weight(weight, q_n, q_p):
    if -q_n != q_p:
        return np.where(weight > 0., weight, weight / q_n * q_p)
    else:
        return np.absolute(weight)


def get_threshold(model, slw_rate=0.0):
    all = []
    state = model.state_dict()
    for name, item in model.named_parameters():
        if 'mask' in name:
            key = name.replace('quantizer.mask', 'weight')
            weight = state[key].data.view(-1).cpu().numpy()
            #importance = norm_weight(weight, module.q_n, module.q_p)
            importance = np.abs(weight)
            all.append(importance)
    
    all = np.concatenate(all)
    threshold = np.sort(all)[int(all.shape[0] * (1. - slw_rate))]
    return threshold


def update(model, threshold):
    state = model.state_dict()
    for name, item in model.named_parameters():
        if 'mask' in name:
            key = name.replace('quantizer.mask', 'weight')
            new_mask = torch.where(state[key].abs() < threshold, torch.ones_like(item), torch.zeros_like(item))
            item.data.copy_(new_mask)


def cal_cent_prob_dynamic_mask(model):
    mask_nonzeros = 0
    mask_length = 0

    for name, item in model.module.named_parameters():
        if 'mask' in name:
            mask_nonzeros += item.sum().cpu().numpy()
            mask_length += item.numel()

    cent_prob = mask_nonzeros / mask_length
    return cent_prob


def cal_cent_prob_static_mask(model):
    nb_total = 0
    nb_tw = 0

    state = model.state_dict()
    for name, item in model.named_parameters():
        if 'mask' in name:
            w = state[name.replace('quantizer.mask', 'weight')]
            s = state[name.replace('mask', 'step')]
            m = item

            nb_total += m.numel()
            nb_tw += m.sum()
            nb_tw += torch.where((w.abs() < s.abs() * 1.5) & (m == 0), torch.ones_like(m), torch.zeros_like(m)).sum()

    cent_prob = nb_tw / nb_total
    return cent_prob


def cal_cent_prob_step(model):
    nb_total = 0
    nb_slw = 0

    for name, module in model.named_modules():
        if type(module) in [QuantConv2d, QuantLinear]:
            if module.centralize:
                x = module.weight
                s = module.quantizer.step_abs

                nb_total += module.weight.numel()
                nb_slw += torch.where(x.abs() < s * 1.5, torch.zeros_like(x), torch.ones_like(x)).sum()

    cent_prob = 1. - nb_slw / nb_total
    return cent_prob
