#
# Faster R-CNN in PyTorch and TensorFlow 2 w/ Keras
# tf2/FasterRCNN/datasets/training_sample.py
# Copyright 2021-2022 Bart Trzynadlowski
#
# Definition of a training sample. All items pertaining to a single sample are
# bundled together for convenience.
#

from dataclasses import dataclass
import numpy as np
from PIL import Image
from typing import List
from typing import Tuple


@dataclass
class Box:
  class_index: int
  class_name: str
  corners: np.ndarray
  
  def __repr__(self):
    return "[class=%s (%f,%f,%f,%f)]" % (self.class_name, self.corners[0], self.corners[1], self.corners[2], self.corners[3])

  def __str__(self):
    return repr(self)

@dataclass
class TrainingSample:
  anchor_map:                 np.ndarray                # shape (feature_map_height,feature_map_width,num_anchors*4), with each anchor as [center_y,center_x,height,width]
  anchor_valid_map:           np.ndarray                # shape (feature_map_height,feature_map_width,num_anchors), indicating which anchors are valid (do not cross image boundaries)
  gt_rpn_map:                 np.ndarray                # TODO: describe me
  gt_rpn_object_indices:      List[Tuple[int,int,int]]  # list of (y,x,k) coordinates of anchors in gt_rpn_map that are labeled as object
  gt_rpn_background_indices:  List[Tuple[int,int,int]]  # list of (y,x,k) coordinates of background anchors
  gt_boxes:                   List[Box]                 # list of ground-truth boxes, scaled
  image_data:                 np.ndarray                # shape (3,height,width), pre-processed and scaled to size expected by model
  image:                      Image                     # PIL image data (for debug rendering), scaled
  filepath:                   str                       # file path of image
