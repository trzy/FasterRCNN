#
# Explanation of RoI pooling: https://towardsdatascience.com/understanding-region-of-interest-part-1-roi-pooling-e4f5dd65bb44
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
  def __init__(self, pool_size, num_rois, **kwargs):
    self.pool_size = pool_size
    self.num_rois = num_rois
    super().__init__(**kwargs)

  def get_config(self):
    config = {
      "pool_size": self.pool_size,
      "num_rois": self.num_rois
    }
    base_config = super(RoIPoolingLayer, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def compute_output_shape(self, input_shape):
    map_shape, rois_shape = input_shape
    assert map_shape[0] == rois_shape[0]
    num_samples = map_shape[0]
    num_channels = map_shape[3]
    return (num_samples, self.num_rois, self.pool_size, self.pool_size, num_channels)

  def build(self, input_shape):
   #  assert len(input_shape) == 2 and len(input_shape[0]) == 4 and len(input_shape[1]) == 3
    self._num_channels = input_shape[0][3]

  def call(self, inputs):
#    x_map = inputs[0]
#    x_roi = inputs[1]
#
#    # When defining model, x_map.shape[0] will be None because we don't have a batch size.
#    # Using tf.shape() creates a dynamic scalar tensor that points to the batch size, and
#    # will be evaluated when it is known. See: https://github.com/tensorflow/tensorflow/issues/31991
#    batch_size = tf.shape(x_map)[0]
#
#
#    #print((batch_size, self.num_rois, self.pool_size, self.pool_size, self._num_channels))
#    return tf.ones(dtype = tf.float32, shape = (batch_size, self.num_rois, self.pool_size, self.pool_size, self._num_channels))

  #TODO: save the note about tf.shape(x_map)[0]

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