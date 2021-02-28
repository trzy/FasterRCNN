import numpy as np
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.layers import Layer

class RoIPoolingLayer(Layer):
  """
  Input shape:
    Two tensors [x_map, x_roi] each with shape:
      x_map: (samples, height, width, channels)
      x_roi: (samples, num_rois, 4), where ROIs have the ordering (x, y, width, height)
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
    x_map = inputs[0]
    x_roi = inputs[1]

    # When defining model, x_map.shape[0] will be None because we don't have a batch size.
    # Using tf.shape() creates a dynamic scalar tensor that points to the batch size, and
    # will be evaluated when it is known. See: https://github.com/tensorflow/tensorflow/issues/31991
    batch_size = tf.shape(x_map)[0]

    print((batch_size, self.num_rois, self.pool_size, self.pool_size, self._num_channels))
    return tf.ones(dtype = tf.float32, shape = (batch_size, self.num_rois, self.pool_size, self.pool_size, self._num_channels))

