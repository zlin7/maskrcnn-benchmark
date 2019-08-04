# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .lr_scheduler import WarmupMultiStepLR, WarmupResetEpochStepLR


def make_optimizer(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if cfg.SOLVER.MTEHOD == 'SGD':
        optimizer = torch.optim.SGD(params, lr, momentum=cfg.SOLVER.MOMENTUM)
    else:
        optimizer = torch.optim.Adam(params, lr)
    return optimizer


def make_lr_scheduler(cfg, optimizer):
    if cfg.SOLVER.WARMUP_METHOD == 'warmup_reset_per_epoch':
        return WarmupResetEpochStepLR(optimizer,
                                      gamma=cfg.SOLVER.GAMMA,
                                      warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
                                      warmup_iters=cfg.SOLVER.WARMUP_ITERS,
                                      )

    return WarmupMultiStepLR(
        optimizer,
        cfg.SOLVER.STEPS,
        cfg.SOLVER.GAMMA,
        warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
        warmup_iters=cfg.SOLVER.WARMUP_ITERS,
        warmup_method=cfg.SOLVER.WARMUP_METHOD,
    )
