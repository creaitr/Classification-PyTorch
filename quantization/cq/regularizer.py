import os
import time
import math

import numpy as np
import torch

from .qnn import *


def cal_curr_lmbd(target, epoch, epochs):
    if epoch < epochs // 2:
        return target * (1 - (1 - epoch / epochs * 2) ** 0.5)
    else:
        return target


def regularize(cfg, epoch, model):
    losses = []
    for name, module in model.named_modules():
        if type(module) in [QuantConv2d, QuantLinear]:
            if module.centralize:
                losses.append(cal_pl1(module.weight, module.quantizer.step_abs, module.q_p))
    loss = torch.stack(losses).sum()
    #reg_lmbd = cal_curr_lmbd(cfg.reg_lmbd, epoch, cfg.epochs)
    reg_lmbd = cfg.reg_lmbd
    return reg_lmbd * loss


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


def cal_pl1(weight, step, q_p):
    #with torch.no_grad():
    weight_abs = weight.abs()

    step_grad_scale = 1.0 / ((q_p * weight.numel()) ** 0.5)
    step = grad_scale(step, step_grad_scale)
    #with torch.no_grad():
    threshold = step * 1.0

    loss = torch.where(weight_abs < threshold, torch.zeros_like(weight), (weight_abs - threshold)).sum()
    return loss


def cal_curr_lmbd2(target, epoch, epochs):
    # if epoch < 5:
    #     return 1e-6 + (target - 1e-6) * (epoch / 5)
    # else:
    #     return target
    if epoch < epochs / 3:
        return target * (1 - (1 - epoch / epochs * 3) ** 0.5)
    else:
        return target


def cal_curr_slw_rate(cfg, target, epoch, epochs):
    if cfg.bitw == 3:
        init_slw_rate = 0.3
    elif cfg.bitw == 4:
        init_slw_rate = 0.5
    elif cfg.bitw == 5:
        init_slw_rate = 0.7
    
    if epoch < 5:
        return init_slw_rate + (target - init_slw_rate) * (epoch / 5)
    else:
        return target


def regularize_th(cfg, epoch, model):
    #slw_rate = cal_curr_slw_rate(cfg, cfg.slw_rate, epoch, cfg.epochs)
    #th = get_threshold(model, slw_rate)
    th = get_threshold(model, cfg.slw_rate)

    losses = []
    for name, module in model.named_modules():
        if type(module) in [QuantConv2d, QuantLinear]:
            if module.centralize:
                losses.append(cal_pl1_th(module.weight, module.quantizer.step_abs, module.q_p, th))
    loss = torch.stack(losses).sum()
    reg_lmbd = cal_curr_lmbd2(cfg.reg_lmbd, epoch, cfg.epochs)
    #reg_lmbd = cfg.reg_lmbd
    return reg_lmbd * loss


def get_threshold(model, slw_rate=0.0):
    with torch.no_grad():
        all = []
        for name, module in model.named_modules():
            if type(module) in [QuantConv2d, QuantLinear]:
                if module.centralize:
                    weight = module.weight.data.view(-1)
                    importance = weight.abs()
                    all.append(importance)
        
        all = torch.cat(all)
        sorted, idx = all.sort(descending=True)
        threshold = sorted[int(all.size(0) * slw_rate)]
    return threshold


def cal_pl1_th(weight, step, q_p, th):
    #with torch.no_grad():
    weight_abs = weight.abs()

    step_grad_scale = 1.0 / ((q_p * weight.numel()) ** 0.5)
    step = grad_scale(step, step_grad_scale)
    #with torch.no_grad():
    threshold = step * 1.0

    loss = torch.where((weight_abs > threshold) & (weight_abs < th), (weight_abs - threshold), torch.zeros_like(weight)).sum()
    return loss


# 403 def WR(x):
# 404     param_name = x.op.name
# 405     if 'conv1' not in param_name and 'fct' not in param_name:
# 406         name_scope, device = x.op.name.split('/W')
# 407         if 'tower' in name_scope:
# 408             name_scope = name_scope[name_scope.index('/') + 1:]
# 409 
# 410         with tf.variable_scope(name_scope, reuse=tf.AUTO_REUSE):
# 411             s = tf.stop_gradient(tf.get_variable('sw'))
# 412 
# 413         pL1 = tf.reduce_sum(tf.clip_by_value(tf.abs(x), s * 1.5, 10000.) - s * 1.5) * (s * 1.5)
# 414         #lmbd = 0.2 / tf.math.sqrt(tf.cast(tf.size(x), tf.float32))
# 415         #lmbd = lmbd * (2 ** (BITW - 1) - 1) * s
# 416         return pL1 * 1.
# 417     else:
# 418         return tf.zeros_like(x)

# 421 def NR(sw):
# 422     param_name = sw.op.name
# 423     if 'conv1' not in param_name and 'fct' not in param_name:
# 424         name_scope, device = sw.op.name.split('/sw')
# 425         if 'tower' in name_scope:
# 426             name_scope = name_scope[name_scope.index('/') + 1:]
# 427 
# 428         with tf.variable_scope(name_scope, reuse=tf.AUTO_REUSE):
# 429             x = tf.stop_gradient(tf.get_variable('W'))
# 430 
# 431         L2 = tf.multiply(0.5, tf.square(sw))
# 432 
# 433         prob = tf.reduce_mean(tf.where(tf.abs(x) < sw * 1.5, tf.zeros_like(x), tf.ones_like(x)))
# 434 
# 435         nb_x = tf.math.sqrt(tf.cast(tf.size(x), tf.float32))
# 436 
# 437         return -1. * prob * nb_x * L2 * 1.
# 438     else:
# 439         return tf.zeros_like(sw)
