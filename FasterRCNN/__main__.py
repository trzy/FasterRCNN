#
# Faster R-CNN in Keras: https://towardsdatascience.com/faster-r-cnn-object-detection-implemented-by-keras-for-custom-data-from-googles-open-images-125f62b9141a
# Understanding RoI pooling: https://towardsdatascience.com/understanding-region-of-interest-part-1-roi-pooling-e4f5dd65bb44
# NMS for object detection: https://towardsdatascience.com/non-maximum-suppression-nms-93ce178e177c

from . import utils
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
  options = parser.parse_args()

  from .Models import vgg16

  from tensorflow.keras import Model
  from tensorflow.keras import Input


  conv_model = vgg16.conv_layers(input_shape=(600,800,3))

  model = Model([conv_model.input], conv_model.outputs)
  model.summary()

  print(conv_model.input)

  print("Loading VOC dataset...")
  voc = VOC(dataset_dir = options.dataset_dir, scale = 600)

  #print(voc.get_boxes_per_image_path(dataset = "val"))
  