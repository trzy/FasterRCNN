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

  #voc = VOC(dataset_dir = options.dataset_dir)
  #print(voc.index_to_class_name)


  from tensorflow.keras import Model
  from tensorflow.keras import Input
  from .Models.RoIPoolingLayer import RoIPoolingLayer

  # Build a model to test just the RoI layer using the Keras functional API
  input_map = Input(shape = (5,5,1))
  input_rois = Input(shape = (4,4))
  output_roi_pool = RoIPoolingLayer(pool_size = 7, num_rois = 4)([input_map, input_rois])
  roi_model = Model([input_map, input_rois], output_roi_pool)
  roi_model.summary()

  # Run the model on a test input
  test_input_map = np.eye(5).reshape(5,5,1)
  test_input_rois = np.array([[0,0,1,1], [0,0,1,2], [2,2,1,1], [4,4,1,1]])

  x_maps = np.array([ test_input_map, test_input_map ])
  x_rois = np.array([ test_input_rois, test_input_rois ])

  x = [ x_maps, x_rois ]
  y = roi_model.predict(x = x)
  print(y.shape)