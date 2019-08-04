# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

#from ..backbone import build_backbone
import maskrcnn_benchmark.modeling.backbone.resnet as mod_resnet
from collections import OrderedDict
import torch.nn.functional as F
import ipdb


def build_resnet_backbone(cfg):
    body = mod_resnet.ResNet(cfg)
    model = nn.Sequential(OrderedDict([("body", body)]))
    model.out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    return model

class PretrainCNN(nn.Module):

    def __init__(self, cfg):
        super(PretrainCNN, self).__init__()
        self.backbone = build_resnet_backbone(cfg)
        #self.backbone = build_backbone(cfg)
        #
        #@registry.BACKBONES.register("R-50-C4")
        #@registry.BACKBONES.register("R-50-C5")

        #self.rpn = build_rpn(cfg, self.backbone.out_channels)
        #self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)
        #self.fc = nn.Linear(self.backbone.out_channels * 7, 2)
        self.batch_norm = nn.BatchNorm1d(4096)
        self.fc = nn.Linear(4096, 256)
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, images, targets=None, weight=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        features_flat = torch.cat([torch.flatten(x, 1) for x in features])
        features_flat = self.batch_norm(features_flat)

        features1 = F.relu_(self.fc(features_flat))
        features2 = F.relu_(self.fc1(features1))
        pred = F.relu_(self.fc2(features2))

        softmax = F.softmax(pred, dim=1)
        #ipdb.set_trace()
        if weight is not None:
            loss = F.binary_cross_entropy(softmax, targets, weight=weight)
        else:
            loss = F.binary_cross_entropy(softmax, targets)
        return {"loss_classifier": loss}, softmax
        if self.training:
            loss = F.binary_cross_entropy(softmax, targets)
            return {"loss_classifier":loss}
        return softmax
