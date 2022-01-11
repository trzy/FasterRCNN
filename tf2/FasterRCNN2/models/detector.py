
import sys 
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import models
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import glorot_normal

from .roi_pooling_layer import RoIPoolingLayer


def layers(image_shape, feature_map, proposals, num_classes, custom_roi_pool, detector_class_activations, l2):
  assert len(feature_map.shape) == 4

  regularizer = tf.keras.regularizers.l2(l2)

  class_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.01)
  regressor_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.001)

  # RoI pool layer creates 7x7 map for each proposal. These are independently
  # passed through two fully-connected layers.
  if custom_roi_pool:
    # Convert proposals from image-space (y1, x1, y2, x2) to feature map space
    # (y1, x1, height, width)
    proposals = tf.cast(proposals, dtype = tf.int32)                  # RoIs must be integral for RoIPoolingLayer
    map_dimensions = tf.shape(feature_map)[1:3]                       # (batches, height, width, channels) -> (height, width)
    map_limits = tf.tile(map_dimensions, multiples = [2]) - 1         # (height, width, height, width)
    roi_corners = tf.minimum(proposals // 16, map_limits)             # to feature map space and clamp against map edges
    roi_corners = tf.maximum(roi_corners, 0)
    roi_dimensions = roi_corners[:,2:4] - roi_corners[:,0:2] + 1
    rois = tf.concat([ roi_corners[:,0:2], roi_dimensions ], axis = 1)  # (N,4), where each row is (y1, x2, height, width) in feature map units
    rois = tf.expand_dims(rois, axis = 0)                             # (1,N,4), batch size of 1, as expected by RoIPoolingLayer

    # Pool
    pool = RoIPoolingLayer(pool_size = 7, name = "roi_pool")([feature_map, rois])
  else:
    # Convert to normalized RoIs with each coordinate in [0,1]
    rois = proposals / [ image_shape[1], image_shape[2], image_shape[1], image_shape[2] ]

    # https://github.com/kevinjliang/tf-Faster-RCNN/blob/master/Lib/roi_pool.py
    # Crop and resize to 14x14 and then max pool
    num_rois = tf.shape(rois)[0];
    region = tf.image.crop_and_resize(image = feature_map, boxes = rois, box_indices = tf.zeros(num_rois, dtype = tf.int32), crop_size = [14, 14])
    pool = tf.nn.max_pool(region, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")
    pool = tf.expand_dims(pool, axis = 0) # (num_rois, 7, 7, 512) -> (1, num_rois, 7, 7, 512)
 
  # Fully-connected layers act as classifiers as in VGG-16 and use the same
  # layer names so that they can be pre-initialized with VGG-16 weights
  flattened = TimeDistributed(Flatten())(pool)
  fc1 = TimeDistributed(name = "fc1", layer = Dense(units = 4096, activation = "relu", kernel_regularizer = regularizer))(flattened)
  fc2 = TimeDistributed(name = "fc2", layer = Dense(units = 4096, activation = "relu", kernel_regularizer = regularizer))(fc1)
  out = fc2

  # Output: classifier
  class_activation = "softmax" if detector_class_activations else None
  classifier = TimeDistributed(name = "classifier_class", layer = Dense(units = num_classes, activation = class_activation, kernel_initializer = class_initializer))(out)

  # Output: box regressions. Unique regression weights for each possible class
  # excluding background class, hence the use of (num_classes-1). Class index 1
  # regressions are therefore at indices: 0*4:0*4+1.
  regressor = TimeDistributed(name = "classifier_boxes", layer = Dense(units = 4 * (num_classes - 1), activation = "linear", kernel_initializer = regressor_initializer))(out)

  return classifier, regressor

def class_loss(y_predicted, y_true, from_logits):
  """
  Keras implementation of classifier network classification loss. The inputs
  are shaped (M,N,C), where M is the number of batches (i.e., 1), N is the
  number of proposed RoIs to classify, and C is the number of classes. One-hot
  encoding is used, hence categorical crossentropy.
  """
  scale_factor = 1.0
  N = tf.cast(tf.shape(y_true)[1], dtype = tf.float32) + K.epsilon()  # number of proposals
  if from_logits:
    #def do_log(x):
    #  y_predicted = x[0]
    #  y_true = x[1]
    #  loss = K.mean(K.categorical_crossentropy(target = y_true, output = y_predicted, from_logits = True))
    #  tf.print("loss=", loss, "y_predicted=", y_predicted, output_stream = "file:///projects/FasterRCNN/tf2/out.txt", summarize = -1)
    #  return y_predicted
    #y_predicted = Lambda(do_log)((y_predicted, y_true))
    return scale_factor * K.sum(K.categorical_crossentropy(target = y_true, output = y_predicted, from_logits = True)) / N
  else:
    return scale_factor * K.sum(K.categorical_crossentropy(y_true, y_predicted)) / N

def regression_loss(y_predicted, y_true):
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

  # TODO:
  # Not clear which of these methods of normalization are ideal, or whether it
  # even matters
  #N = tf.reduce_sum(y_mask) / 4.0 + K.epsilon()                      # N = number of positive boxes
  N = tf.cast(tf.shape(y_true)[1], dtype = tf.float32) + K.epsilon() # N = number of proposals
  #N = tf.reduce_sum(y_mask) + K.epsilon()                             # N = number of parameters (i.e., number of positive boxes * 4)
  relevant_loss_terms = y_mask * losses
  return scale_factor * K.sum(relevant_loss_terms) / N
