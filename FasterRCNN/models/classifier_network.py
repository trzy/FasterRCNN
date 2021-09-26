from . import region_proposal_network
from .roi_pooling_layer import RoIPoolingLayer

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import models
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import TimeDistributed

def layers(num_classes, input_map, proposal_boxes, dropout_fraction, l2 = 0, weight_initializer = "normal"):
  """
  Constructs classifier network from conv. net output map and proposal boxes
  scaled to the map's coordinate space. Shape of proposal_boxes is (N,4) where
  N is the number of proposals and the second dimension contains:

    0: y_min
    1: x_min
    2: y_max
    3: x_max
  """
  assert len(input_map.shape) == 4
  
  regularizer = tf.keras.regularizers.l2(l2)

  if weight_initializer == "normal":
    class_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.01)
    regressor_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.001)
  else:
    class_initializer = weight_initializer
    regressor_initializer = weight_initializer

  # RoI pool layer creates 7x7 map for each proposal. These are independently
  # passed through two fully-connected layers.
  pool = RoIPoolingLayer(pool_size = 7, name = "roi_pool")([input_map, proposal_boxes])
  flattened = TimeDistributed(Flatten())(pool)
  
  # Fully-connected layers act as classifiers as in VGG-16 and use the same
  # layer names so that they can be pre-initialized with VGG-16 weights
  if dropout_fraction != 0:
    fc1 = TimeDistributed(name = "fc1", layer = Dense(units = 4096, activation = "relu", kernel_regularizer = regularizer))(flattened)
    do1 = TimeDistributed(Dropout(dropout_fraction))(fc1)
    fc2 = TimeDistributed(name = "fc2", layer = Dense(units = 4096, activation = "relu", kernel_regularizer = regularizer))(do1)
    do2 = TimeDistributed(Dropout(dropout_fraction))(fc2)
    out = do2
  else:
    fc1 = TimeDistributed(name = "fc1", layer = Dense(units = 4096, activation = "relu", kernel_regularizer = regularizer))(flattened)
    fc2 = TimeDistributed(name = "fc2", layer = Dense(units = 4096, activation = "relu", kernel_regularizer = regularizer))(fc1)
    out = fc2

  # Output: classifier
  classifier = TimeDistributed(name = "classifier_class", layer = Dense(units = num_classes, activation = "softmax", kernel_initializer = class_initializer))(out)

  # Output: box regressions. Unique regression weights for each possible class
  # excluding background class, hence the use of (num_classes-1). Class index 1
  # regressions are therefore at indices: 0*4:0*4+1.
  regressor = TimeDistributed(name = "classifier_boxes", layer = Dense(units = 4 * (num_classes - 1), activation = "linear", kernel_initializer = regressor_initializer))(out)

  return [ classifier, regressor ]
