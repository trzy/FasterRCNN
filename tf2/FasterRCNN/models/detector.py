#
# Faster R-CNN in PyTorch and TensorFlow 2 w/ Keras
# tf2/FasterRCNN/models/detector.py
# Copyright 2021-2022 Bart Trzynadlowski
#
# Tensorflow/Keras implementation of the final detector stage of Faster R-CNN.
# As input, takes a series of proposals (or RoIs) and produces classifications
# and boxes. The boxes are parameterized as modifications to the original
# incoming proposal boxes. That is, the proposal boxes are exactly analogous to
# the anchors that the RPN stage uses.
#

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import models
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras import backend as K

from .roi_pooling_layer import RoIPoolingLayer


class DetectorNetwork(tf.keras.Model):
  def __init__(self, num_classes, custom_roi_pool, activate_class_outputs, l2, dropout_probability):
    super().__init__()

    self._num_classes = num_classes
    self._activate_class_outputs = activate_class_outputs
    self._dropout_probability = dropout_probability

    regularizer = tf.keras.regularizers.l2(l2)
    class_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.01)
    regressor_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.001)

    # If custom_roi_pool flag is set, we use our custom implementation,
    # otherwise, tf operations that can approximate the operation will be used
    # in call().
    self._roi_pool = RoIPoolingLayer(pool_size = 7, name = "custom_roi_pool") if custom_roi_pool else None

    # Fully-connected layers with optional dropout. Initial weights will be
    # loaded from pre-trained VGG-16 ImageNet model by parent Faster R-CNN
    # module. These layers act as classifiers as in VGG-16 and use the same
    # names as Keras' built-in implementation of VGG-16. TimeDistributed() is
    # used to iterate over the proposal dimension and apply the layer to each
    # of the proposals.
    self._flatten = TimeDistributed(Flatten())
    self._fc1 = TimeDistributed(name = "fc1", layer = Dense(units = 4096, activation = "relu", kernel_regularizer = regularizer))
    self._dropout1 = TimeDistributed(Dropout(dropout_probability))
    self._fc2 = TimeDistributed(name = "fc2", layer = Dense(units = 4096, activation = "relu", kernel_regularizer = regularizer))
    self._dropout2 = TimeDistributed(Dropout(dropout_probability))

    # Output: classifier
    class_activation = "softmax" if activate_class_outputs else None
    self._classifier = TimeDistributed(name = "classifier_class", layer = Dense(units = num_classes, activation = class_activation, kernel_initializer = class_initializer))

    # Output: box delta regressions. Unique regression weights for each
    # possible class excluding background class, hence the use of
    # (num_classes-1). Class index 1 regressions are therefore at
    # indices: 0*4:0*4+1.
    self._regressor = TimeDistributed(name = "classifier_boxes", layer = Dense(units = 4 * (num_classes - 1), activation = "linear", kernel_initializer = regressor_initializer))

  def call(self, inputs, training):
    # Unpack inputs
    input_image = inputs[0]
    feature_map = inputs[1]
    proposals = inputs[2]
    assert len(feature_map.shape) == 4

    # RoI pooling: creates a 7x7 map for each proposal (1, num_rois, 7, 7, 512)
    if self._roi_pool:
      # Use our custom layer. Need to convert proposals from image-space
      # (y1, x1, y2, x2) to feature map space (y1, x1, height, width).
      proposals = tf.cast(proposals, dtype = tf.int32)                  # RoIs must be integral for RoIPoolingLayer
      map_dimensions = tf.shape(feature_map)[1:3]                       # (batches, height, width, channels) -> (height, width)
      map_limits = tf.tile(map_dimensions, multiples = [2]) - 1         # (height, width, height, width)
      roi_corners = tf.minimum(proposals // 16, map_limits)             # to feature map space and clamp against map edges
      roi_corners = tf.maximum(roi_corners, 0)
      roi_dimensions = roi_corners[:,2:4] - roi_corners[:,0:2] + 1
      rois = tf.concat([ roi_corners[:,0:2], roi_dimensions ], axis = 1)  # (N,4), where each row is (y1, x2, height, width) in feature map units
      rois = tf.expand_dims(rois, axis = 0)                             # (1,N,4), batch size of 1, as expected by RoIPoolingLayer
      pool = RoIPoolingLayer(pool_size = 7, name = "roi_pool")([feature_map, rois])
    else:
      # Crop the proposals, resize to 14x14 (with bilinear interpolation) and
      # max pool down to 7x7. This works just as well and is used in several
      # TensorFlow implementations of Faster R-CNN, such as:
      # https://github.com/kevinjliang/tf-Faster-RCNN/blob/master/Lib/roi_pool.py

      # Convert to normalized RoIs with each coordinate in [0,1]
      image_height = tf.shape(input_image)[1] # height in pixels
      image_width = tf.shape(input_image)[2]  # width in pixels
      rois = proposals / [ image_height, image_width, image_height, image_width ]

      # Crop, resize, pool
      num_rois = tf.shape(rois)[0];
      region = tf.image.crop_and_resize(image = feature_map, boxes = rois, box_indices = tf.zeros(num_rois, dtype = tf.int32), crop_size = [14, 14])
      pool = tf.nn.max_pool(region, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")
      pool = tf.expand_dims(pool, axis = 0) # (num_rois, 7, 7, 512) -> (1, num_rois, 7, 7, 512)
          
    # Pass through final layers
    flattened = self._flatten(pool)
    if training and self._dropout_probability != 0:
      fc1 = self._fc1(flattened)
      do1 = self._dropout1(fc1)
      fc2 = self._fc2(do1)
      do2 = self._dropout2(fc2)
      out = do2
    else:
      fc1 = self._fc1(flattened)
      fc2 = self._fc2(fc1)
      out = fc2 
    class_activation = "softmax" if self._activate_class_outputs else None
    classes = self._classifier(out)
    box_deltas = self._regressor(out)

    return [ classes, box_deltas ]

  @staticmethod
  def class_loss(y_predicted, y_true, from_logits):
    """
    Computes detector network classification loss.

    Parameters
    ----------
    y_predicted : tf.Tensor
      Class predictions, shaped (1, N, num_classes), where N is the number of
      detections (i.e., the number of proposals fed into the detector network).
    y_true : tf.Tensor
      Ground truth, shaped (1, N, num_classes). One-hot-encoded labels.
    from_logits : bool
      If true, y_predicted is given as logits (that is, softmax was not
      applied), otherwise, as probability scores (softmax applied).

    Returns
    -------
    tf.Tensor
      Scalar loss.
    """
    scale_factor = 1.0
    N = tf.cast(tf.shape(y_true)[1], dtype = tf.float32) + K.epsilon()  # number of proposals
    if from_logits:
      return scale_factor * K.sum(K.categorical_crossentropy(target = y_true, output = y_predicted, from_logits = True)) / N
    else:
      return scale_factor * K.sum(K.categorical_crossentropy(y_true, y_predicted)) / N
  
  @staticmethod
  def regression_loss(y_predicted, y_true):
    """
    Computes detector network box delta regression loss.

    Parameters
    ----------
    y_predicted : tf.Tensor
      Predicted box delta regressions in parameterized form (ty, tx, th, tw).
      Shaped (1, N, 4 * (num_classes - 1)). Class 0 (background) obviously has
      no box associated with it.
    y_true : tf.Tensor
      Ground truth box delta regression targets, shaped
      (1, N, 2, 4 * (num_classes - 1)). Elements [:,:,0,:] are masks indicating
      which of the regression targets [:,:,1,:] to use for the given proposal.
      That is, [0,n,0,:] is an array of 1 or 0 indicating which of [0,n,1,:]
      are valid for inclusion in the loss. For non-background proposals, there
      will be 4 unmasked values corresponding to (ty, tx, th, tw).
    
    Returns
    -------
    tf.Tensor
      Scalar loss.
    """
    scale_factor = 1.0
    sigma = 1.0
    sigma_squared = sigma * sigma
  
    # We want to unpack the regression targets and the mask of valid targets into
    # tensors each of the same shape as the predicted: 
    #   (batch_size, num_proposals, 4*(num_classes-1))
    # y_true has shape:
    #   (batch_size, num_proposals, 2, 4*(num_classes-1))
    y_mask = y_true[:,:,0,:]
    y_true_targets = y_true[:,:,1,:]
  
    # Compute element-wise loss using robust L1 function for all 4 regression
    # targets
    x = y_true_targets - y_predicted
    x_abs = tf.math.abs(x)
    is_negative_branch = tf.stop_gradient(tf.cast(tf.less(x_abs, 1.0 / sigma_squared), dtype = tf.float32))
    R_negative_branch = 0.5 * x * x * sigma_squared
    R_positive_branch = x_abs - 0.5 / sigma_squared
    losses = is_negative_branch * R_negative_branch + (1.0 - is_negative_branch) * R_positive_branch
  
    # Accumulate the relevant terms and normalize by the number of proposals
    N = tf.cast(tf.shape(y_true)[1], dtype = tf.float32) + K.epsilon()  # N = number of proposals
    relevant_loss_terms = y_mask * losses
    return scale_factor * K.sum(relevant_loss_terms) / N
