# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.structures.keypoint import PersonKeypoints
import os

class DEEPSZDataset():
    DATA_DIR = "/media/zhen/Data/deepsz/maps/reso0.25"
    LABEL_DIR = "/media/zhen/Data/deepsz/maps/annotations/0.25/z0.25_mvir2e14"
    def __init__(self, data_dir = DATA_DIR, label_dir=LABEL_DIR, component="skymap"):

        self.data_dir = os.path.join(data_dir, component)
        self.label_dir = label_dir

    def __getitem__(self, idx):
        pass

    def get_img_info(self, idx):

        pass