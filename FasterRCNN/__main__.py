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

# good test images:
# 2010_004041.jpg
# 2010_005080.jpg
if __name__ == "__main__":
  parser = argparse.ArgumentParser("FasterRCNN")
  parser.add_argument("--dataset-dir", metavar = "path", type = str, action = "store", default = "\\projects\\voc\\vocdevkit\\voc2012", help = "Dataset directory")
  parser.add_argument("--show-image", metavar = "file", type = str, action = "store", help = "Show an image with ground truth and corresponding anchor boxes")
  options = parser.parse_args()

  print("Loading VOC dataset...")
  voc = VOC(dataset_dir = options.dataset_dir, scale = 600)

  
  from .models import vgg16
  from .models import region_proposal_network

  from tensorflow.keras import Model
  from tensorflow.keras import Input

  if options.show_image:
    info = voc.get_image_description(path = voc.get_full_path(options.show_image))

    # Need to build the model for this image size in order to be able to visualize boxes correctly
    conv_model = vgg16.conv_layers(input_shape = (info.height,info.width,3))
    classifier_output, regression_output = region_proposal_network.layers(input_map = conv_model.outputs[0])
    model = Model([conv_model.input], [classifier_output, regression_output])

    print(classifier_output.shape, model.input.shape)
    
    visualization.show_annotated_image(voc = voc, filename = options.show_image, draw_anchor_intersections = True, image_input_map = model.input, anchor_map = classifier_output)

  train_data = voc.train_data()
  for i in range(voc.num_samples["train"]):
    image_path, (ground_truth_regressions, positive_anchors, negative_anchors) = next(train_data)
    print(len(positive_anchors))

