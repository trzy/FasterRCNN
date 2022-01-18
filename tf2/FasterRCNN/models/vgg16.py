#
# Faster R-CNN in PyTorch and TensorFlow 2 w/ Keras
# tf2/FasterRCNN/models/vgg16.py
# Copyright 2021-2022 Bart Trzynadlowski
#
# TensorFlow/Keras implementation of the VGG-16 backbone for use as a feature
# extractor in Faster R-CNN. Only the convolutional layers are used.
#

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import models
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.initializers import glorot_normal


class FeatureExtractor(tf.keras.Model):
  def __init__(self, l2 = 0):
    super().__init__()

    initial_weights = glorot_normal()
    regularizer = tf.keras.regularizers.l2(l2)
    input_shape = (None, None, 3)
  
    # First two convolutional blocks are frozen (not trainable)
    self._block1_conv1 = Conv2D(name = "block1_conv1", input_shape = input_shape, kernel_size = (3,3), strides = 1, filters = 64, padding = "same", activation = "relu", kernel_initializer = initial_weights, trainable = False)
    self._block1_conv2 = Conv2D(name = "block1_conv2", kernel_size = (3,3), strides = 1, filters = 64, padding = "same", activation = "relu", kernel_initializer = initial_weights, trainable = False)
    self._block1_maxpool = MaxPooling2D(pool_size = 2, strides = 2)

    self._block2_conv1 = Conv2D(name = "block2_conv1", kernel_size = (3,3), strides = 1, filters = 128, padding = "same", activation = "relu", kernel_initializer = initial_weights, trainable = False)
    self._block2_conv2 = Conv2D(name = "block2_conv2", kernel_size = (3,3), strides = 1, filters = 128, padding = "same", activation = "relu", kernel_initializer = initial_weights, trainable = False)
    self._block2_maxpool = MaxPooling2D(pool_size = 2, strides = 2)

    # Weight decay begins from these layers onward: https://github.com/rbgirshick/py-faster-rcnn/blob/master/models/pascal_voc/VGG16/faster_rcnn_end2end/train.prototxt
    self._block3_conv1 = Conv2D(name = "block3_conv1", kernel_size = (3,3), strides = 1, filters = 256, padding = "same", activation = "relu", kernel_initializer = initial_weights, kernel_regularizer = regularizer)
    self._block3_conv2 = Conv2D(name = "block3_conv2", kernel_size = (3,3), strides = 1, filters = 256, padding = "same", activation = "relu", kernel_initializer = initial_weights, kernel_regularizer = regularizer)
    self._block3_conv3 = Conv2D(name = "block3_conv3", kernel_size = (3,3), strides = 1, filters = 256, padding = "same", activation = "relu", kernel_initializer = initial_weights, kernel_regularizer = regularizer)
    self._block3_maxpool = MaxPooling2D(pool_size = 2, strides = 2)

    self._block4_conv1 = Conv2D(name = "block4_conv1", kernel_size = (3,3), strides = 1, filters = 512, padding = "same", activation = "relu", kernel_initializer = initial_weights, kernel_regularizer = regularizer)
    self._block4_conv2 = Conv2D(name = "block4_conv2", kernel_size = (3,3), strides = 1, filters = 512, padding = "same", activation = "relu", kernel_initializer = initial_weights, kernel_regularizer = regularizer)
    self._block4_conv3 = Conv2D(name = "block4_conv3", kernel_size = (3,3), strides = 1, filters = 512, padding = "same", activation = "relu", kernel_initializer = initial_weights, kernel_regularizer = regularizer)
    self._block4_maxpool = MaxPooling2D(pool_size = 2, strides = 2)

    self._block5_conv1 = Conv2D(name = "block5_conv1", kernel_size = (3,3), strides = 1, filters = 512, padding = "same", activation = "relu", kernel_initializer = initial_weights, kernel_regularizer = regularizer)
    self._block5_conv2 = Conv2D(name = "block5_conv2", kernel_size = (3,3), strides = 1, filters = 512, padding = "same", activation = "relu", kernel_initializer = initial_weights, kernel_regularizer = regularizer)
    self._block5_conv3 = Conv2D(name = "block5_conv3", kernel_size = (3,3), strides = 1, filters = 512, padding = "same", activation = "relu", kernel_initializer = initial_weights, kernel_regularizer = regularizer)

  def call(self, input_image):
    y = self._block1_conv1(input_image)
    y = self._block1_conv2(y)
    y = self._block1_maxpool(y)

    y = self._block2_conv1(y)
    y = self._block2_conv2(y)
    y = self._block2_maxpool(y)

    y = self._block3_conv1(y)
    y = self._block3_conv2(y)
    y = self._block3_conv3(y)
    y = self._block3_maxpool(y)

    y = self._block4_conv1(y)
    y = self._block4_conv2(y)
    y = self._block4_conv3(y)
    y = self._block4_maxpool(y)

    y = self._block5_conv1(y)
    y = self._block5_conv2(y)
    y = self._block5_conv3(y)

    return y
