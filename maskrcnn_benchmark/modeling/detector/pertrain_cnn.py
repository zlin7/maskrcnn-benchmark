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
        self.fc = nn.Linear(4096, 2)

    def forward(self, images, targets=None):
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
        features2 = self.fc(features_flat)
        softmax = F.softmax(features2, dim=0)
        if self.training:
            loss = F.binary_cross_entropy(softmax, targets)
            return {"loss_classifier":loss}
        return softmax

        proposals, proposal_losses = self.rpn(images, features, targets)
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result
