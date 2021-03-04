#
# Faster R-CNN in Keras: https://towardsdatascience.com/faster-r-cnn-object-detection-implemented-by-keras-for-custom-data-from-googles-open-images-125f62b9141a
# Understanding RoI pooling: https://towardsdatascience.com/understanding-region-of-interest-part-1-roi-pooling-e4f5dd65bb44
# NMS for object detection: https://towardsdatascience.com/non-maximum-suppression-nms-93ce178e177c

from . import utils
from . import visualization
from .dataset import VOC

import argparse
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras
import time

if __name__ == "__main__":
  parser = argparse.ArgumentParser("FasterRCNN")
  parser.add_argument("--dataset-dir", metavar = "path", type = str, action = "store", default = "\\projects\\voc\\vocdevkit\\voc2012", help = "Dataset directory")
  parser.add_argument("--show-image", metavar = "file", type = str, action = "store", help = "Show an image with ground truth and corresponding anchor boxes")
  options = parser.parse_args()

  print("Loading VOC dataset...")
  voc = VOC(dataset_dir = options.dataset_dir, scale = 600)

  if options.show_image:
    visualization.show_annotated_image(voc = voc, filename = options.show_image)

  
  from .models import vgg16
  from .models import region_proposal_network

  from tensorflow.keras import Model
  from tensorflow.keras import Input


  conv_model = vgg16.conv_layers(input_shape=(709,600,3))
  classifier_output, regression_output = region_proposal_network.layers(input_map = conv_model.outputs[0], anchors_per_location = 9)

  model = Model([conv_model.input], [classifier_output, regression_output])
  model.summary()

  print(conv_model.input)

  print( region_proposal_network.convert_anchor_coordinate_from_rpn_layer_to_image_space(y = 100, x = 100, image_input_map = conv_model.input, anchor_map = classifier_output) )

  #print("Loading VOC dataset...")
  #voc = VOC(dataset_dir = options.dataset_dir, scale = 600)

  #print(voc.get_boxes_per_image_path(dataset = "val"))
  