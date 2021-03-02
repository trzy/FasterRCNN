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

#  # Build a model to test just the RoI layer using the Keras functional API
#  input_map = Input(shape = (5,5,1))
#  input_rois = Input(shape = (4,4))
#  output_roi_pool = RoIPoolingLayer(pool_size = 7, num_rois = 4)([input_map, input_rois])
#  roi_model = Model([input_map, input_rois], output_roi_pool)
#  roi_model.summary()
#
#  # Run the model on a test input
#  test_input_map = np.eye(5).reshape(5,5,1)
#  test_input_rois = np.array([[0,0,1,1], [0,0,1,2], [2,2,1,1], [4,4,1,1]])
#
#  x_maps = np.array([ test_input_map, test_input_map ])
#  x_rois = np.array([ test_input_rois, test_input_rois ])
#
#  x = [ x_maps, x_rois ]
#  y = roi_model.predict(x = x)
#  print(y.shape)

  test_map = np.array([
    [ 1, 2, 3, 4, 5, 6, 7, 8 ],
    [ 2, 3, 4, 5, 6, 7, 8, 9 ],
    [ 3, 4, 5, 6, 7, 8, 9, 1 ],
    [ 4, 5, 6, 7, 8, 9, 1, 2 ],
    [ 5, 6, 7, 8, 9, 1, 2, 3 ],
    [ 6, 7, 8, 9, 1, 2, 3, 4 ],
    [ 7, 8, 9, 1, 2, 3, 4, 5 ],
    [ 8, 10, 1, 2, 3, 4, 11, 6 ],
    [ 9, 1, 2, 3, 4, 5, 6, 7 ],
  ])
  test_map2 = np.array([
    [ 1, 2, 3, 4, 5, 6, 7, 8 ],
    [ 2, 3, 4, 5, 6, 7, 8, 9 ],
    [ 3, 4, 5, 6, 7, 8, 9, 1 ],
    [ 4, 5, 6, 7, 8, 9, 1, 2 ],
    [ 5, 6, 7, 8, 9, 1, 2, 3 ],
    [ 6, 7, 8, 9, 1, 2, 3, 4 ],
    [ 7, 8, 9, 1, 2, 3, 4, 5 ],
    [ 8, 12, 1, 2, 3, 4, 11, 6 ],
    [ 9, 1, 2, 3, 4, 5, 6, 7 ],
  ])
  test_map = np.stack([test_map,test_map2], axis=2)  # create 2 chgannels
  test_map = tf.constant(test_map, dtype = tf.float32)

  pool_size = tf.constant(2)
  num_channels = tf.constant(2)

  region_height = tf.constant(5, dtype = tf.int32)
  region_width = tf.constant(7, dtype = tf.int32)
  region_of_interest = tf.slice(test_map, tf.constant([3, 0, 0]), [region_height, region_width, num_channels]) # extract RoI from full map


  x_step = tf.cast(region_width, dtype = tf.float32) / tf.cast(pool_size, dtype = tf.float32)
  y_step = tf.cast(region_height, dtype = tf.float32) / tf.cast(pool_size, dtype = tf.float32)



  #
  # Create 2D tensors corresponding to the pool dimensions with the
  # x index (or y index) in each element by constructing a tensor of
  # stacked x ranges (or concatenated y ranges). E.g., if pool_size is 3:
  #
  #     0 1 2
  # x = 0 1 2
  #     0 1 2
  #
  #     0 0 0
  # y = 1 1 1
  #     2 2 2
  #
  # The multiplication here should not be interpreted as a matrix multiply!
  #
  y_range_int = tf.reshape(tf.range(pool_size), shape = (pool_size, 1))
  x_range_int = tf.range(pool_size)
  y_range = tf.cast(y_range_int, dtype = tf.float32)
  x_range = tf.cast(x_range_int, dtype = tf.float32)
  y = tf.ones(shape = (pool_size, pool_size), dtype = tf.float32) * y_range
  x = tf.ones(shape = (pool_size, pool_size), dtype = tf.float32) * x_range

  #
  # Compute the final start and end positions using the following logic:
  #
  #   x_start = int(x * x_step)
  #   x_end = int((x + 1) * x_step) if (x + 1) < pool_size else region_width
  #   y_start = int(y * y_step)
  #   y_end = int((y + 1) * y_step) if (y + 1) < pool_size else region_height
  #
  y_start = tf.cast(y * y_step, dtype = tf.int32)
  x_start = tf.cast(x * x_step, dtype = tf.int32)

  not_last_y = tf.less((y + 1), tf.cast(pool_size, dtype = tf.float32))
  y_end = tf.where(not_last_y, tf.cast((y + 1) * y_step, dtype = tf.int32), region_height)

  not_last_x = tf.less((x + 1), tf.cast(pool_size, dtype = tf.float32))
  x_end = tf.where(not_last_x, tf.cast((x + 1) * x_step, dtype = tf.int32), region_width)


  print(y_start)
  print(x_start)
  print(y_end)
  print(x_end)

  #
  # The x_start/x_end and y_start/y_end matrices define the extents of the
  # cells in the source feature map to take the max of, yielding a tensor
  # of shape (pool_size, pool_size).
  #
  # That is, for any i, the cell is defined by the ranges:
  #
  #   y_start[i] : y_end[i]
  #   x_start[i] : x_end[i]
  #
  # Where the start is inclusive and the end is exclusive.
  #

  #TODO: we probably could have computed x_size/y_size directly: int(x_step) or int(x_step)+1
  y_size = y_end - y_start
  x_size = x_end - x_start

  @tf.function
  def do_compute(region_of_interest, pool_y_start, pool_x_start, y_step, x_step, region_height, region_width, pool_size, num_channels):
    pool_y_start_int = tf.cast(pool_y_start, dtype = tf.int32)
    pool_x_start_int = tf.cast(pool_x_start, dtype = tf.int32)
    y_start = tf.cast(pool_y_start * y_step, dtype = tf.int32)
    x_start = tf.cast(pool_x_start * x_step, dtype = tf.int32)
    y_end = tf.cond((pool_y_start_int + 1) < pool_size,
      lambda: tf.cast((pool_y_start + 1) * y_step, dtype = tf.int32),
      lambda: region_height
    )
    x_end = tf.cond((pool_x_start_int + 1) < pool_size,
      lambda: tf.cast((pool_x_start + 1) * x_step, dtype = tf.int32),
      lambda: region_width
    )
    y_size = y_end - y_start
    x_size = x_end - x_start
    print(pool_y_start)
    print(pool_x_start)
    pool_cell = tf.slice(region_of_interest, [y_start, x_start, 0], [y_size, x_size, num_channels])
    return tf.math.reduce_max(pool_cell, axis=(1,0))  # keep channels independent

  x_range = tf.cast(tf.range(pool_size), dtype = tf.float32)
  y_range = tf.cast(tf.range(pool_size), dtype = tf.float32)
  cells = tf.map_fn(
    fn = lambda y: tf.map_fn(
      fn = lambda x:
        do_compute(region_of_interest, pool_y_start = y, pool_x_start = x, y_step = y_step, x_step = x_step, region_height = region_height, region_width = region_width, pool_size = pool_size, num_channels = num_channels),
      elems = x_range
    ),
    elems = y_range
  )



  print(cells)
  print(cells[0,0,0], cells[0,1,0])
  print(cells[1,0,0], cells[1,1,0])

  print(cells[0,0,1], cells[0,1,1])
  print(cells[1,0,1], cells[1,1,1])
  #cells = tf.map_fn(y_range,
  #  lambda y: tf.map_fn(x_range,
  #    lambda x: compute_dest_cell_range_in_src_map(dest_y_start = y, dest_x_start = x, src_y_step = y_step, src_x_step = x_step, src_height = region_height, src_width = region_width, pool_size = pool_size)
  #  )
  #)
  #print(region_of_interest)
  #print(cells)


  x = tf.constant([1,2,3])
  y = tf.constant([4,5,6])
  #c = tf.stack([x, y], axis=1)
  c=tf.constant([x,y])
  #c = tf.constant([[1,2,3],[4,5,6]])
  print(c)
  z=tf.map_fn(
    fn = lambda x: x[0] + x[1],
    elems = c
  )
  print(z)