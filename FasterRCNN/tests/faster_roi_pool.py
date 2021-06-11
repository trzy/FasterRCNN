#
# FasterRCNN for Keras
# Copyright 2021 Bart Trzynadlowski
#
# tests/faster_roi_pool.py
#
# Script to test the speed of the RoI pooling layer. Useful for testing new
# optimizations.
#

from ..models.roi_pooling_layer import RoIPoolingLayer

import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import Input
import time

def generate_random_data(num_channels):
  # Generate random feature map
  width = random.randint(map_limits[0], map_limits[1])
  height = random.randint(map_limits[0], map_limits[1])
  feature_map = np.random.rand(height, width, num_channels)

  # Generate random ROIs within feature map
  rois = np.zeros((num_rois, 4))
  for j in range(num_rois):
    rois[j,0] = random.randint(0, height - 1)         # y
    rois[j,1] = random.randint(0, width - 1)          # x
    rois[j,2] = random.randint(1, height - rois[j,0]) # height
    rois[j,3] = random.randint(1, width - rois[j,1])  # width

  return feature_map, rois

if __name__ == "__main__":
  pool_size = 7
  num_channels = 512
  num_rois = 64                               # fixed number of RoIs
  input_map = Input(shape = (None,None,512))  # input map size: variable width and height
  input_rois = Input(shape = (num_rois,4), dtype = tf.int32)
  output_roi_pool = RoIPoolingLayer(pool_size = pool_size)([input_map, input_rois])
  roi_model = Model([input_map, input_rois], output_roi_pool)
  roi_model.summary()

  # Run several iterations with randomized inputs
  map_limits = (15, 55)
  timings = []
  for i in range(100):
    # Generate random data and expand batch size to 1
    feature_map, rois = generate_random_data(num_channels = num_channels) 
    feature_map = np.expand_dims(feature_map, axis = 0)
    rois = np.expand_dims(rois, axis = 0)

    # Run layer
    t0 = time.perf_counter()
    roi_model.predict_on_batch(x = [ feature_map, rois ])
    t = time.perf_counter() - t0
    timings.append(t)

  # Average and max times
  print("Max time: %1.2f" % max(timings))
  print("Avg time: %1.2f" % np.mean(timings))
