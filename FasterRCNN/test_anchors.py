from .models.roi_pooling_layer import RoIPoolingLayer
from .models import vgg16
from .models.region_proposal_network import _compute_anchor_sizes
from .models.region_proposal_network import compute_all_anchor_boxes

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import Input


if __name__ == "__main__":
  width = 800
  height = 600
  img_shape = (600,800,3)

  anchor_boxes, anchor_boxes_valid = compute_all_anchor_boxes(input_image_shape = img_shape)
  anchors = anchor_boxes.reshape((anchor_boxes.shape[0] * anchor_boxes.shape[1] * 9, 4))
  for i in range(len(anchors)):
    cy, cx, h, w = anchors[i]
    print("%d, %d -- %d x %d" % (cx, cy, w, h))
  
