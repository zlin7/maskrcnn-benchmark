# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from bisect import bisect_right

import torch


# FIXME ideally this would be achieved with a CombinedLRScheduler,
# separating MultiStepLR with WarmupLR
# but the current LRScheduler design doesn't allow it
class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        #print(self.last_epoch, warmup_factor, warmup_factor * self.gamma ** bisect_right(self.milestones, self.last_epoch))
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


import ipdb
class WarmupResetEpochStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_factor=1.0 / 3,
        warmup_iters=20,
        warmup_method="linear",
        last_epoch=-1,
        last_iter_this_epoch=-1,
        gamma=0.1,
    ):
        #if not list(milestones) == sorted(milestones):
        #    raise ValueError(
        #        "Milestones should be a list of" " increasing integers. Got {}",
        #        milestones,
        #    )

        if warmup_method not in ("linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        #self.milestones = milestones
        #self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        #self.warmup_method = warmup_method
        self.last_iter_this_epoch = last_iter_this_epoch
        self.initial_lrs = []
        super(WarmupResetEpochStepLR, self).__init__(optimizer, last_epoch)
        self.initial_lrs = [lr for lr in self.base_lrs]

    def get_lr(self):
        warmup_factor = 1
        if self.last_iter_this_epoch < self.warmup_iters:
            alpha = float(self.last_iter_this_epoch) / self.warmup_iters
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        #print(self.base_lrs)
        #ipdb.set_trace()
        print(self.last_iter_this_epoch, warmup_factor)
        return [
            base_lr
            * warmup_factor
            for base_lr in self.base_lrs
        ]

    def step_iter(self, epoch, curr_iter_this_epoch):
        self.last_iter_this_epoch = curr_iter_this_epoch
        self.step(epoch)
    def step_wrap(self, epoch=None):
        self.step(epoch)
