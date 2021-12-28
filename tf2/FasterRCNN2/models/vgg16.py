import numpy as np
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.initializers import glorot_normal
from tensorflow.keras.optimizers import SGD

def conv_layers(input_shape = (None,None,3), l2 = 0):
  initial_weights = glorot_normal()
  regularizer = tf.keras.regularizers.l2(l2)
  
  model = Sequential()

  model.add( Conv2D(name = "block1_conv1", input_shape = input_shape, kernel_size = (3,3), strides = 1, filters = 64, padding = "same", activation = "relu", kernel_initializer = initial_weights) )
  model.add( Conv2D(name = "block1_conv2", kernel_size = (3,3), strides = 1, filters = 64, padding = "same", activation = "relu", kernel_initializer = initial_weights) )
  model.add( MaxPooling2D(pool_size = 2, strides = 2) )

  model.add( Conv2D(name = "block2_conv1", kernel_size = (3,3), strides = 1, filters = 128, padding = "same", activation = "relu", kernel_initializer = initial_weights) )
  model.add( Conv2D(name = "block2_conv2", kernel_size = (3,3), strides = 1, filters = 128, padding = "same", activation = "relu", kernel_initializer = initial_weights) )
  model.add( MaxPooling2D(pool_size = 2, strides = 2) )

  model.add( Conv2D(name = "block3_conv1", kernel_size = (3,3), strides = 1, filters = 256, padding = "same", activation = "relu", kernel_initializer = initial_weights, kernel_regularizer = regularizer) )  # weight decay begins at this layer: https://github.com/rbgirshick/py-faster-rcnn/blob/master/models/pascal_voc/VGG16/faster_rcnn_end2end/train.prototxt
  model.add( Conv2D(name = "block3_conv2", kernel_size = (3,3), strides = 1, filters = 256, padding = "same", activation = "relu", kernel_initializer = initial_weights, kernel_regularizer = regularizer) )
  model.add( Conv2D(name = "block3_conv3", kernel_size = (3,3), strides = 1, filters = 256, padding = "same", activation = "relu", kernel_initializer = initial_weights, kernel_regularizer = regularizer) )
  model.add( MaxPooling2D(pool_size = 2, strides = 2) )

  model.add( Conv2D(name = "block4_conv1", kernel_size = (3,3), strides = 1, filters = 512, padding = "same", activation = "relu", kernel_initializer = initial_weights, kernel_regularizer = regularizer) )
  model.add( Conv2D(name = "block4_conv2", kernel_size = (3,3), strides = 1, filters = 512, padding = "same", activation = "relu", kernel_initializer = initial_weights, kernel_regularizer = regularizer) )
  model.add( Conv2D(name = "block4_conv3", kernel_size = (3,3), strides = 1, filters = 512, padding = "same", activation = "relu", kernel_initializer = initial_weights, kernel_regularizer = regularizer) )
  model.add( MaxPooling2D(pool_size = 2, strides = 2) )

  model.add( Conv2D(name = "block5_conv1", kernel_size = (3,3), strides = 1, filters = 512, padding = "same", activation = "relu", kernel_initializer = initial_weights, kernel_regularizer = regularizer) )
  model.add( Conv2D(name = "block5_conv2", kernel_size = (3,3), strides = 1, filters = 512, padding = "same", activation = "relu", kernel_initializer = initial_weights, kernel_regularizer = regularizer) )
  model.add( Conv2D(name = "block5_conv3", kernel_size = (3,3), strides = 1, filters = 512, padding = "same", activation = "relu", kernel_initializer = initial_weights, kernel_regularizer = regularizer) )

  return model

def load_imagenet_weights(model):
  keras_model = tf.keras.applications.VGG16(weights = "imagenet")
  for keras_layer in keras_model.layers:
    weights = keras_layer.get_weights()
    if len(weights) > 0:
      our_layer = [ layer for layer in model.layers if layer.name == keras_layer.name ]
      if len(our_layer) > 0:
        print("Loading VGG-16 ImageNet weights into layer: %s" % our_layer[0].name)
        our_layer[0].set_weights(weights)

def compute_feature_map_shape(input_image_shape):
  """
  Returns the 2D shape of the VGG-16 output map (height, width), which will be
  1/16th of the input image for VGG-16.
  """
  return (input_image_shape[0] // 16, input_image_shape[1] // 16)
