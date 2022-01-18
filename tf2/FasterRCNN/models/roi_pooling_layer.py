#
# Faster R-CNN in PyTorch and TensorFlow 2 w/ Keras
# tf2/FasterRCNN/models/roi_pooling_layer.py
# Copyright 2021-2022 Bart Trzynadlowski
#
# Custom implementation of RoI pooling for TensorFlow/Keras. TensorFlow lacks
# an RoI pooling operation that is exactly analogous to Faster R-CNN's, so I
# attempted to implement my own. Performance is abysmal so this is not
# recommended for actual use and is left here as an experiment. It was found
# that unrolling the map functions yielded a slight improvement in performance.
# This was done with unroll_roi_pool.py.
#
# Explanation of RoI pooling:
# https://towardsdatascience.com/understanding-region-of-interest-part-1-roi-pooling-e4f5dd65bb44
#

import numpy as np
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.layers import Layer


class RoIPoolingLayer(Layer):
  """
  Input shape:
    Two tensors [x_maps, x_rois] each with shape:
      x_maps: (samples, height, width, channels), representing the feature maps for this batch, of type tf.float32
      x_rois: (samples, num_rois, 4), where RoIs have the ordering (y, x, height, width), all tf.int32
  Output shape:
    (samples, num_rois, pool_size, pool_size, channels)
  """
  def __init__(self, pool_size, **kwargs):
    self.pool_size = pool_size
    super().__init__(**kwargs)

  def get_config(self):
    config = {
      "pool_size": self.pool_size,
    }
    base_config = super(RoIPoolingLayer, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def compute_output_shape(self, input_shape):
    map_shape, rois_shape = input_shape
    assert len(map_shape) == 4 and len(rois_shape) == 3 and rois_shape[2] == 4
    assert map_shape[0] == rois_shape[0]  # same number of samples
    num_samples = map_shape[0]
    num_channels = map_shape[3]
    num_rois = rois_shape[1]
    return (num_samples, num_rois, self.pool_size, self.pool_size, num_channels)

  def call(self, inputs):
    #
    # Unused here but useful to know:
    #
    # When defining model, x_map.shape[0] will be None because we don't have a batch size.
    # Using tf.shape() creates a dynamic scalar tensor that points to the batch size, and
    # will be evaluated when it is known. See: https://github.com/tensorflow/tensorflow/issues/31991
    #
    #   x_map = inputs[0]
    #   batch_size = tf.shape(x_map)[0]
    #

    #
    # Inputs are a list, [ x_maps, x_rois ], where x_maps and x_rois must have
    # the same batch size, N. The first application of map_fn() iterates over
    # N samples of [ x_map, x_roi ]. For this to work, the data type of the
    # final tensor must be specified otherwise map_fn() apparently infers a
    # very different (and incorrect) input element.
    #
    # This basically iterates over every sample in the batch. This is the
    # outer-most of a pair of map_fn() iterations and stacks its results
    # in the batch dimension, (samples, ...).
    #
    if self.pool_size == 7 and inputs[0].shape[3] == 512:
      # Special case optimization: 7x7x512 pools, ~4-5x speed-up
      return tf.map_fn(
        fn = lambda input_pair:
          RoIPoolingLayer._compute_pooled_rois_7x7x512(feature_map = input_pair[0], rois = input_pair[1]),
        elems = inputs,
        fn_output_signature = tf.float32  # this is absolutely required else the fn type inference seems to fail spectacularly
      )
    else:
      # Generic case capable of handling any pool shape
      return tf.map_fn(
        fn = lambda input_pair:
          RoIPoolingLayer._compute_pooled_rois(feature_map = input_pair[0], rois = input_pair[1], pool_size = self.pool_size),
        elems = inputs,
        fn_output_signature = tf.float32  # this is absolutely required else the fn type inference seems to fail spectacularly
      )

  @tf.function
  def _compute_pooled_rois(feature_map, rois, pool_size):
    #
    # Given a feature map and its associated RoIs, iterate over all RoIs for
    # this map. This is the second level of iteration and yields the num_rois
    # dimension: (samples, num_rois, ...)
    #
    return tf.map_fn(
      fn = lambda roi:
        RoIPoolingLayer._compute_pooled_roi(feature_map = feature_map, roi = roi, pool_size = pool_size),
      elems = rois,
      fn_output_signature = tf.float32
    )

  @tf.function
  def _compute_pooled_roi(feature_map, roi, pool_size):
    #
    # Given a feature map and a single RoI, computes the pooled map of shape
    # (pool_size, pool_size).
    #

    # Crop out the region of interest from the feature map
    region_y = roi[0]
    region_x = roi[1]
    region_height = roi[2]
    region_width = roi[3]
    num_channels = feature_map.shape[2]
    region_of_interest = tf.slice(feature_map, [region_y, region_x, 0], [region_height, region_width, num_channels])

    # Compute step size within the region of interest (feature map)
    x_step = tf.cast(region_width, dtype = tf.float32) / tf.cast(pool_size, dtype = tf.float32)
    y_step = tf.cast(region_height, dtype = tf.float32) / tf.cast(pool_size, dtype = tf.float32)

    #
    # Compute the pooled map for this RoI having shape (pool_size, pool_size).
    # This is done by a nested iteration with x being the inner, fast index and
    # y being the outer, slow index, resulting in shape (size_y, size_x), where
    # both sizes here are pool_size.
    #
    x_range = tf.cast(tf.range(pool_size), dtype = tf.float32)
    y_range = tf.cast(tf.range(pool_size), dtype = tf.float32)
    pooled_cells = tf.map_fn(
      fn = lambda y: tf.map_fn(
        fn = lambda x:
          RoIPoolingLayer._pool_one_cell(region_of_interest, pool_y_start = y, pool_x_start = x, y_step = y_step, x_step = x_step, region_height = region_height, region_width = region_width, pool_size = pool_size, num_channels = num_channels),
        elems = x_range
      ),
      elems = y_range
    )
    return pooled_cells

  @tf.function
  def _pool_one_cell(region_of_interest, pool_y_start, pool_x_start, y_step, x_step, region_height, region_width, pool_size, num_channels):
    #
    # This function maps a single pooling cell over some part of the RoI and
    # then computes the max of the RoI cells inside that pooling cell. The
    # operation is performed per-channel, yielding a result of shape
    # (1, num_channels).
    #
    # Compute the start and end positions using the following logic:
    #
    #   x_start = int(x * x_step)
    #   x_end = int((x + 1) * x_step) if (x + 1) < pool_size else region_width
    #   y_start = int(y * y_step)
    #   y_end = int((y + 1) * y_step) if (y + 1) < pool_size else region_height
    #
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

    # Extract this cell from the region and return the max
    y_size = tf.math.maximum(y_end - y_start, 1)  # if RoI is smaller than pool area, y_end - y_start can be less than 1 (0); we want to sample at least one cell
    x_size = tf.math.maximum(x_end - x_start, 1)
    pool_cell = tf.slice(region_of_interest, [y_start, x_start, 0], [y_size, x_size, num_channels])
    return tf.math.reduce_max(pool_cell, axis=(1,0))  # keep channels independent

  @tf.function
  def _compute_pooled_rois_7x7x512(feature_map, rois):
    # Special case: 7x7x512, unrolled pool width and height (7x7=49)
    return tf.map_fn(
      fn = lambda roi: tf.reshape(
        tf.stack([
          # y=0,x=0
          tf.math.reduce_max(
            tf.slice(
              feature_map[ roi[0]:roi[0]+roi[2], roi[1]:roi[1]+roi[3], 0:512 ],
              [
                tf.cast(0 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32),
                tf.cast(0 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32),
                0
              ],
              [
                tf.math.maximum(1, tf.cast((0 + 1) * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(0 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32)),
                tf.math.maximum(1, tf.cast((0 + 1) * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(0 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32)),
                512
              ]
            ),
            axis = (1,0)
          ),
          # y=0,x=1
          tf.math.reduce_max(
            tf.slice(
              feature_map[ roi[0]:roi[0]+roi[2], roi[1]:roi[1]+roi[3], 0:512 ],
              [
                tf.cast(0 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32),
                tf.cast(1 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32),
                0
              ],
              [
                tf.math.maximum(1, tf.cast((0 + 1) * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(0 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32)),
                tf.math.maximum(1, tf.cast((1 + 1) * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(1 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32)),
                512
              ]
            ),
            axis = (1,0)
          ),
          # y=0,x=2
          tf.math.reduce_max(
            tf.slice(
              feature_map[ roi[0]:roi[0]+roi[2], roi[1]:roi[1]+roi[3], 0:512 ],
              [
                tf.cast(0 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32),
                tf.cast(2 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32),
                0
              ],
              [
                tf.math.maximum(1, tf.cast((0 + 1) * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(0 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32)),
                tf.math.maximum(1, tf.cast((2 + 1) * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(2 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32)),
                512
              ]
            ),
            axis = (1,0)
          ),
          # y=0,x=3
          tf.math.reduce_max(
            tf.slice(
              feature_map[ roi[0]:roi[0]+roi[2], roi[1]:roi[1]+roi[3], 0:512 ],
              [
                tf.cast(0 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32),
                tf.cast(3 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32),
                0
              ],
              [
                tf.math.maximum(1, tf.cast((0 + 1) * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(0 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32)),
                tf.math.maximum(1, tf.cast((3 + 1) * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(3 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32)),
                512
              ]
            ),
            axis = (1,0)
          ),
          # y=0,x=4
          tf.math.reduce_max(
            tf.slice(
              feature_map[ roi[0]:roi[0]+roi[2], roi[1]:roi[1]+roi[3], 0:512 ],
              [
                tf.cast(0 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32),
                tf.cast(4 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32),
                0
              ],
              [
                tf.math.maximum(1, tf.cast((0 + 1) * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(0 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32)),
                tf.math.maximum(1, tf.cast((4 + 1) * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(4 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32)),
                512
              ]
            ),
            axis = (1,0)
          ),
          # y=0,x=5
          tf.math.reduce_max(
            tf.slice(
              feature_map[ roi[0]:roi[0]+roi[2], roi[1]:roi[1]+roi[3], 0:512 ],
              [
                tf.cast(0 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32),
                tf.cast(5 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32),
                0
              ],
              [
                tf.math.maximum(1, tf.cast((0 + 1) * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(0 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32)),
                tf.math.maximum(1, tf.cast((5 + 1) * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(5 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32)),
                512
              ]
            ),
            axis = (1,0)
          ),
          # y=0,x=6
          tf.math.reduce_max(
            tf.slice(
              feature_map[ roi[0]:roi[0]+roi[2], roi[1]:roi[1]+roi[3], 0:512 ],
              [
                tf.cast(0 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32),
                tf.cast(6 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32),
                0
              ],
              [
                tf.math.maximum(1, tf.cast((0 + 1) * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(0 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32)),
                tf.math.maximum(1, roi[3] - tf.cast(6 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32)),
                512
              ]
            ),
            axis = (1,0)
          ),
          # y=1,x=0
          tf.math.reduce_max(
            tf.slice(
              feature_map[ roi[0]:roi[0]+roi[2], roi[1]:roi[1]+roi[3], 0:512 ],
              [
                tf.cast(1 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32),
                tf.cast(0 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32),
                0
              ],
              [
                tf.math.maximum(1, tf.cast((1 + 1) * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(1 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32)),
                tf.math.maximum(1, tf.cast((0 + 1) * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(0 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32)),
                512
              ]
            ),
            axis = (1,0)
          ),
          # y=1,x=1
          tf.math.reduce_max(
            tf.slice(
              feature_map[ roi[0]:roi[0]+roi[2], roi[1]:roi[1]+roi[3], 0:512 ],
              [
                tf.cast(1 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32),
                tf.cast(1 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32),
                0
              ],
              [
                tf.math.maximum(1, tf.cast((1 + 1) * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(1 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32)),
                tf.math.maximum(1, tf.cast((1 + 1) * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(1 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32)),
                512
              ]
            ),
            axis = (1,0)
          ),
          # y=1,x=2
          tf.math.reduce_max(
            tf.slice(
              feature_map[ roi[0]:roi[0]+roi[2], roi[1]:roi[1]+roi[3], 0:512 ],
              [
                tf.cast(1 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32),
                tf.cast(2 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32),
                0
              ],
              [
                tf.math.maximum(1, tf.cast((1 + 1) * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(1 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32)),
                tf.math.maximum(1, tf.cast((2 + 1) * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(2 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32)),
                512
              ]
            ),
            axis = (1,0)
          ),
          # y=1,x=3
          tf.math.reduce_max(
            tf.slice(
              feature_map[ roi[0]:roi[0]+roi[2], roi[1]:roi[1]+roi[3], 0:512 ],
              [
                tf.cast(1 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32),
                tf.cast(3 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32),
                0
              ],
              [
                tf.math.maximum(1, tf.cast((1 + 1) * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(1 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32)),
                tf.math.maximum(1, tf.cast((3 + 1) * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(3 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32)),
                512
              ]
            ),
            axis = (1,0)
          ),
          # y=1,x=4
          tf.math.reduce_max(
            tf.slice(
              feature_map[ roi[0]:roi[0]+roi[2], roi[1]:roi[1]+roi[3], 0:512 ],
              [
                tf.cast(1 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32),
                tf.cast(4 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32),
                0
              ],
              [
                tf.math.maximum(1, tf.cast((1 + 1) * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(1 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32)),
                tf.math.maximum(1, tf.cast((4 + 1) * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(4 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32)),
                512
              ]
            ),
            axis = (1,0)
          ),
          # y=1,x=5
          tf.math.reduce_max(
            tf.slice(
              feature_map[ roi[0]:roi[0]+roi[2], roi[1]:roi[1]+roi[3], 0:512 ],
              [
                tf.cast(1 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32),
                tf.cast(5 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32),
                0
              ],
              [
                tf.math.maximum(1, tf.cast((1 + 1) * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(1 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32)),
                tf.math.maximum(1, tf.cast((5 + 1) * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(5 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32)),
                512
              ]
            ),
            axis = (1,0)
          ),
          # y=1,x=6
          tf.math.reduce_max(
            tf.slice(
              feature_map[ roi[0]:roi[0]+roi[2], roi[1]:roi[1]+roi[3], 0:512 ],
              [
                tf.cast(1 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32),
                tf.cast(6 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32),
                0
              ],
              [
                tf.math.maximum(1, tf.cast((1 + 1) * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(1 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32)),
                tf.math.maximum(1, roi[3] - tf.cast(6 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32)),
                512
              ]
            ),
            axis = (1,0)
          ),
          # y=2,x=0
          tf.math.reduce_max(
            tf.slice(
              feature_map[ roi[0]:roi[0]+roi[2], roi[1]:roi[1]+roi[3], 0:512 ],
              [
                tf.cast(2 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32),
                tf.cast(0 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32),
                0
              ],
              [
                tf.math.maximum(1, tf.cast((2 + 1) * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(2 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32)),
                tf.math.maximum(1, tf.cast((0 + 1) * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(0 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32)),
                512
              ]
            ),
            axis = (1,0)
          ),
          # y=2,x=1
          tf.math.reduce_max(
            tf.slice(
              feature_map[ roi[0]:roi[0]+roi[2], roi[1]:roi[1]+roi[3], 0:512 ],
              [
                tf.cast(2 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32),
                tf.cast(1 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32),
                0
              ],
              [
                tf.math.maximum(1, tf.cast((2 + 1) * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(2 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32)),
                tf.math.maximum(1, tf.cast((1 + 1) * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(1 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32)),
                512
              ]
            ),
            axis = (1,0)
          ),
          # y=2,x=2
          tf.math.reduce_max(
            tf.slice(
              feature_map[ roi[0]:roi[0]+roi[2], roi[1]:roi[1]+roi[3], 0:512 ],
              [
                tf.cast(2 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32),
                tf.cast(2 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32),
                0
              ],
              [
                tf.math.maximum(1, tf.cast((2 + 1) * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(2 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32)),
                tf.math.maximum(1, tf.cast((2 + 1) * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(2 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32)),
                512
              ]
            ),
            axis = (1,0)
          ),
          # y=2,x=3
          tf.math.reduce_max(
            tf.slice(
              feature_map[ roi[0]:roi[0]+roi[2], roi[1]:roi[1]+roi[3], 0:512 ],
              [
                tf.cast(2 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32),
                tf.cast(3 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32),
                0
              ],
              [
                tf.math.maximum(1, tf.cast((2 + 1) * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(2 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32)),
                tf.math.maximum(1, tf.cast((3 + 1) * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(3 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32)),
                512
              ]
            ),
            axis = (1,0)
          ),
          # y=2,x=4
          tf.math.reduce_max(
            tf.slice(
              feature_map[ roi[0]:roi[0]+roi[2], roi[1]:roi[1]+roi[3], 0:512 ],
              [
                tf.cast(2 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32),
                tf.cast(4 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32),
                0
              ],
              [
                tf.math.maximum(1, tf.cast((2 + 1) * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(2 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32)),
                tf.math.maximum(1, tf.cast((4 + 1) * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(4 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32)),
                512
              ]
            ),
            axis = (1,0)
          ),
          # y=2,x=5
          tf.math.reduce_max(
            tf.slice(
              feature_map[ roi[0]:roi[0]+roi[2], roi[1]:roi[1]+roi[3], 0:512 ],
              [
                tf.cast(2 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32),
                tf.cast(5 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32),
                0
              ],
              [
                tf.math.maximum(1, tf.cast((2 + 1) * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(2 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32)),
                tf.math.maximum(1, tf.cast((5 + 1) * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(5 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32)),
                512
              ]
            ),
            axis = (1,0)
          ),
          # y=2,x=6
          tf.math.reduce_max(
            tf.slice(
              feature_map[ roi[0]:roi[0]+roi[2], roi[1]:roi[1]+roi[3], 0:512 ],
              [
                tf.cast(2 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32),
                tf.cast(6 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32),
                0
              ],
              [
                tf.math.maximum(1, tf.cast((2 + 1) * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(2 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32)),
                tf.math.maximum(1, roi[3] - tf.cast(6 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32)),
                512
              ]
            ),
            axis = (1,0)
          ),
          # y=3,x=0
          tf.math.reduce_max(
            tf.slice(
              feature_map[ roi[0]:roi[0]+roi[2], roi[1]:roi[1]+roi[3], 0:512 ],
              [
                tf.cast(3 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32),
                tf.cast(0 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32),
                0
              ],
              [
                tf.math.maximum(1, tf.cast((3 + 1) * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(3 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32)),
                tf.math.maximum(1, tf.cast((0 + 1) * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(0 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32)),
                512
              ]
            ),
            axis = (1,0)
          ),
          # y=3,x=1
          tf.math.reduce_max(
            tf.slice(
              feature_map[ roi[0]:roi[0]+roi[2], roi[1]:roi[1]+roi[3], 0:512 ],
              [
                tf.cast(3 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32),
                tf.cast(1 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32),
                0
              ],
              [
                tf.math.maximum(1, tf.cast((3 + 1) * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(3 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32)),
                tf.math.maximum(1, tf.cast((1 + 1) * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(1 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32)),
                512
              ]
            ),
            axis = (1,0)
          ),
          # y=3,x=2
          tf.math.reduce_max(
            tf.slice(
              feature_map[ roi[0]:roi[0]+roi[2], roi[1]:roi[1]+roi[3], 0:512 ],
              [
                tf.cast(3 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32),
                tf.cast(2 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32),
                0
              ],
              [
                tf.math.maximum(1, tf.cast((3 + 1) * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(3 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32)),
                tf.math.maximum(1, tf.cast((2 + 1) * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(2 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32)),
                512
              ]
            ),
            axis = (1,0)
          ),
          # y=3,x=3
          tf.math.reduce_max(
            tf.slice(
              feature_map[ roi[0]:roi[0]+roi[2], roi[1]:roi[1]+roi[3], 0:512 ],
              [
                tf.cast(3 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32),
                tf.cast(3 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32),
                0
              ],
              [
                tf.math.maximum(1, tf.cast((3 + 1) * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(3 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32)),
                tf.math.maximum(1, tf.cast((3 + 1) * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(3 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32)),
                512
              ]
            ),
            axis = (1,0)
          ),
          # y=3,x=4
          tf.math.reduce_max(
            tf.slice(
              feature_map[ roi[0]:roi[0]+roi[2], roi[1]:roi[1]+roi[3], 0:512 ],
              [
                tf.cast(3 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32),
                tf.cast(4 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32),
                0
              ],
              [
                tf.math.maximum(1, tf.cast((3 + 1) * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(3 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32)),
                tf.math.maximum(1, tf.cast((4 + 1) * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(4 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32)),
                512
              ]
            ),
            axis = (1,0)
          ),
          # y=3,x=5
          tf.math.reduce_max(
            tf.slice(
              feature_map[ roi[0]:roi[0]+roi[2], roi[1]:roi[1]+roi[3], 0:512 ],
              [
                tf.cast(3 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32),
                tf.cast(5 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32),
                0
              ],
              [
                tf.math.maximum(1, tf.cast((3 + 1) * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(3 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32)),
                tf.math.maximum(1, tf.cast((5 + 1) * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(5 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32)),
                512
              ]
            ),
            axis = (1,0)
          ),
          # y=3,x=6
          tf.math.reduce_max(
            tf.slice(
              feature_map[ roi[0]:roi[0]+roi[2], roi[1]:roi[1]+roi[3], 0:512 ],
              [
                tf.cast(3 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32),
                tf.cast(6 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32),
                0
              ],
              [
                tf.math.maximum(1, tf.cast((3 + 1) * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(3 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32)),
                tf.math.maximum(1, roi[3] - tf.cast(6 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32)),
                512
              ]
            ),
            axis = (1,0)
          ),
          # y=4,x=0
          tf.math.reduce_max(
            tf.slice(
              feature_map[ roi[0]:roi[0]+roi[2], roi[1]:roi[1]+roi[3], 0:512 ],
              [
                tf.cast(4 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32),
                tf.cast(0 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32),
                0
              ],
              [
                tf.math.maximum(1, tf.cast((4 + 1) * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(4 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32)),
                tf.math.maximum(1, tf.cast((0 + 1) * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(0 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32)),
                512
              ]
            ),
            axis = (1,0)
          ),
          # y=4,x=1
          tf.math.reduce_max(
            tf.slice(
              feature_map[ roi[0]:roi[0]+roi[2], roi[1]:roi[1]+roi[3], 0:512 ],
              [
                tf.cast(4 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32),
                tf.cast(1 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32),
                0
              ],
              [
                tf.math.maximum(1, tf.cast((4 + 1) * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(4 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32)),
                tf.math.maximum(1, tf.cast((1 + 1) * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(1 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32)),
                512
              ]
            ),
            axis = (1,0)
          ),
          # y=4,x=2
          tf.math.reduce_max(
            tf.slice(
              feature_map[ roi[0]:roi[0]+roi[2], roi[1]:roi[1]+roi[3], 0:512 ],
              [
                tf.cast(4 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32),
                tf.cast(2 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32),
                0
              ],
              [
                tf.math.maximum(1, tf.cast((4 + 1) * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(4 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32)),
                tf.math.maximum(1, tf.cast((2 + 1) * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(2 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32)),
                512
              ]
            ),
            axis = (1,0)
          ),
          # y=4,x=3
          tf.math.reduce_max(
            tf.slice(
              feature_map[ roi[0]:roi[0]+roi[2], roi[1]:roi[1]+roi[3], 0:512 ],
              [
                tf.cast(4 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32),
                tf.cast(3 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32),
                0
              ],
              [
                tf.math.maximum(1, tf.cast((4 + 1) * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(4 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32)),
                tf.math.maximum(1, tf.cast((3 + 1) * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(3 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32)),
                512
              ]
            ),
            axis = (1,0)
          ),
          # y=4,x=4
          tf.math.reduce_max(
            tf.slice(
              feature_map[ roi[0]:roi[0]+roi[2], roi[1]:roi[1]+roi[3], 0:512 ],
              [
                tf.cast(4 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32),
                tf.cast(4 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32),
                0
              ],
              [
                tf.math.maximum(1, tf.cast((4 + 1) * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(4 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32)),
                tf.math.maximum(1, tf.cast((4 + 1) * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(4 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32)),
                512
              ]
            ),
            axis = (1,0)
          ),
          # y=4,x=5
          tf.math.reduce_max(
            tf.slice(
              feature_map[ roi[0]:roi[0]+roi[2], roi[1]:roi[1]+roi[3], 0:512 ],
              [
                tf.cast(4 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32),
                tf.cast(5 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32),
                0
              ],
              [
                tf.math.maximum(1, tf.cast((4 + 1) * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(4 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32)),
                tf.math.maximum(1, tf.cast((5 + 1) * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(5 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32)),
                512
              ]
            ),
            axis = (1,0)
          ),
          # y=4,x=6
          tf.math.reduce_max(
            tf.slice(
              feature_map[ roi[0]:roi[0]+roi[2], roi[1]:roi[1]+roi[3], 0:512 ],
              [
                tf.cast(4 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32),
                tf.cast(6 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32),
                0
              ],
              [
                tf.math.maximum(1, tf.cast((4 + 1) * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(4 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32)),
                tf.math.maximum(1, roi[3] - tf.cast(6 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32)),
                512
              ]
            ),
            axis = (1,0)
          ),
          # y=5,x=0
          tf.math.reduce_max(
            tf.slice(
              feature_map[ roi[0]:roi[0]+roi[2], roi[1]:roi[1]+roi[3], 0:512 ],
              [
                tf.cast(5 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32),
                tf.cast(0 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32),
                0
              ],
              [
                tf.math.maximum(1, tf.cast((5 + 1) * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(5 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32)),
                tf.math.maximum(1, tf.cast((0 + 1) * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(0 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32)),
                512
              ]
            ),
            axis = (1,0)
          ),
          # y=5,x=1
          tf.math.reduce_max(
            tf.slice(
              feature_map[ roi[0]:roi[0]+roi[2], roi[1]:roi[1]+roi[3], 0:512 ],
              [
                tf.cast(5 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32),
                tf.cast(1 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32),
                0
              ],
              [
                tf.math.maximum(1, tf.cast((5 + 1) * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(5 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32)),
                tf.math.maximum(1, tf.cast((1 + 1) * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(1 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32)),
                512
              ]
            ),
            axis = (1,0)
          ),
          # y=5,x=2
          tf.math.reduce_max(
            tf.slice(
              feature_map[ roi[0]:roi[0]+roi[2], roi[1]:roi[1]+roi[3], 0:512 ],
              [
                tf.cast(5 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32),
                tf.cast(2 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32),
                0
              ],
              [
                tf.math.maximum(1, tf.cast((5 + 1) * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(5 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32)),
                tf.math.maximum(1, tf.cast((2 + 1) * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(2 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32)),
                512
              ]
            ),
            axis = (1,0)
          ),
          # y=5,x=3
          tf.math.reduce_max(
            tf.slice(
              feature_map[ roi[0]:roi[0]+roi[2], roi[1]:roi[1]+roi[3], 0:512 ],
              [
                tf.cast(5 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32),
                tf.cast(3 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32),
                0
              ],
              [
                tf.math.maximum(1, tf.cast((5 + 1) * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(5 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32)),
                tf.math.maximum(1, tf.cast((3 + 1) * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(3 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32)),
                512
              ]
            ),
            axis = (1,0)
          ),
          # y=5,x=4
          tf.math.reduce_max(
            tf.slice(
              feature_map[ roi[0]:roi[0]+roi[2], roi[1]:roi[1]+roi[3], 0:512 ],
              [
                tf.cast(5 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32),
                tf.cast(4 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32),
                0
              ],
              [
                tf.math.maximum(1, tf.cast((5 + 1) * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(5 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32)),
                tf.math.maximum(1, tf.cast((4 + 1) * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(4 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32)),
                512
              ]
            ),
            axis = (1,0)
          ),
          # y=5,x=5
          tf.math.reduce_max(
            tf.slice(
              feature_map[ roi[0]:roi[0]+roi[2], roi[1]:roi[1]+roi[3], 0:512 ],
              [
                tf.cast(5 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32),
                tf.cast(5 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32),
                0
              ],
              [
                tf.math.maximum(1, tf.cast((5 + 1) * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(5 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32)),
                tf.math.maximum(1, tf.cast((5 + 1) * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(5 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32)),
                512
              ]
            ),
            axis = (1,0)
          ),
          # y=5,x=6
          tf.math.reduce_max(
            tf.slice(
              feature_map[ roi[0]:roi[0]+roi[2], roi[1]:roi[1]+roi[3], 0:512 ],
              [
                tf.cast(5 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32),
                tf.cast(6 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32),
                0
              ],
              [
                tf.math.maximum(1, tf.cast((5 + 1) * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(5 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32)),
                tf.math.maximum(1, roi[3] - tf.cast(6 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32)),
                512
              ]
            ),
            axis = (1,0)
          ),
          # y=6,x=0
          tf.math.reduce_max(
            tf.slice(
              feature_map[ roi[0]:roi[0]+roi[2], roi[1]:roi[1]+roi[3], 0:512 ],
              [
                tf.cast(6 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32),
                tf.cast(0 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32),
                0
              ],
              [
                tf.math.maximum(1, roi[2] - tf.cast(6 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32)),
                tf.math.maximum(1, tf.cast((0 + 1) * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(0 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32)),
                512
              ]
            ),
            axis = (1,0)
          ),
          # y=6,x=1
          tf.math.reduce_max(
            tf.slice(
              feature_map[ roi[0]:roi[0]+roi[2], roi[1]:roi[1]+roi[3], 0:512 ],
              [
                tf.cast(6 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32),
                tf.cast(1 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32),
                0
              ],
              [
                tf.math.maximum(1, roi[2] - tf.cast(6 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32)),
                tf.math.maximum(1, tf.cast((1 + 1) * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(1 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32)),
                512
              ]
            ),
            axis = (1,0)
          ),
          # y=6,x=2
          tf.math.reduce_max(
            tf.slice(
              feature_map[ roi[0]:roi[0]+roi[2], roi[1]:roi[1]+roi[3], 0:512 ],
              [
                tf.cast(6 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32),
                tf.cast(2 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32),
                0
              ],
              [
                tf.math.maximum(1, roi[2] - tf.cast(6 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32)),
                tf.math.maximum(1, tf.cast((2 + 1) * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(2 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32)),
                512
              ]
            ),
            axis = (1,0)
          ),
          # y=6,x=3
          tf.math.reduce_max(
            tf.slice(
              feature_map[ roi[0]:roi[0]+roi[2], roi[1]:roi[1]+roi[3], 0:512 ],
              [
                tf.cast(6 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32),
                tf.cast(3 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32),
                0
              ],
              [
                tf.math.maximum(1, roi[2] - tf.cast(6 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32)),
                tf.math.maximum(1, tf.cast((3 + 1) * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(3 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32)),
                512
              ]
            ),
            axis = (1,0)
          ),
          # y=6,x=4
          tf.math.reduce_max(
            tf.slice(
              feature_map[ roi[0]:roi[0]+roi[2], roi[1]:roi[1]+roi[3], 0:512 ],
              [
                tf.cast(6 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32),
                tf.cast(4 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32),
                0
              ],
              [
                tf.math.maximum(1, roi[2] - tf.cast(6 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32)),
                tf.math.maximum(1, tf.cast((4 + 1) * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(4 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32)),
                512
              ]
            ),
            axis = (1,0)
          ),
          # y=6,x=5
          tf.math.reduce_max(
            tf.slice(
              feature_map[ roi[0]:roi[0]+roi[2], roi[1]:roi[1]+roi[3], 0:512 ],
              [
                tf.cast(6 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32),
                tf.cast(5 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32),
                0
              ],
              [
                tf.math.maximum(1, roi[2] - tf.cast(6 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32)),
                tf.math.maximum(1, tf.cast((5 + 1) * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32) - tf.cast(5 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32)),
                512
              ]
            ),
            axis = (1,0)
          ),
          # y=6,x=6
          tf.math.reduce_max(
            tf.slice(
              feature_map[ roi[0]:roi[0]+roi[2], roi[1]:roi[1]+roi[3], 0:512 ],
              [
                tf.cast(6 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32),
                tf.cast(6 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32),
                0
              ],
              [
                tf.math.maximum(1, roi[2] - tf.cast(6 * (tf.cast(roi[2], dtype = tf.float32) / 7), dtype = tf.int32)),
                tf.math.maximum(1, roi[3] - tf.cast(6 * (tf.cast(roi[3], dtype = tf.float32) / 7), dtype = tf.int32)),
                512
              ]
            ),
            axis = (1,0)
          ),
        ]),
        shape = (7,7,512)
      ),
      elems = rois,
      fn_output_signature = tf.float32
    )

