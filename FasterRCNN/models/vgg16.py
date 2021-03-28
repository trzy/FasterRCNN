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

  model.add( Conv2D(name = "block1_conv1", input_shape = input_shape, kernel_size = (3,3), strides = 1, filters = 64, padding = "same", activation = "relu", kernel_initializer = initial_weights, kernel_regularizer = regularizer) )
  model.add( Conv2D(name = "block1_conv2", kernel_size = (3,3), strides = 1, filters = 64, padding = "same", activation = "relu", kernel_initializer = initial_weights, kernel_regularizer = regularizer) )
  model.add( MaxPooling2D(pool_size = 2, strides = 2) )

  model.add( Conv2D(name = "block2_conv1", kernel_size = (3,3), strides = 1, filters = 128, padding = "same", activation = "relu", kernel_initializer = initial_weights, kernel_regularizer = regularizer) )
  model.add( Conv2D(name = "block2_conv2", kernel_size = (3,3), strides = 1, filters = 128, padding = "same", activation = "relu", kernel_initializer = initial_weights, kernel_regularizer = regularizer) )
  model.add( MaxPooling2D(pool_size = 2, strides = 2) )

  model.add( Conv2D(name = "block3_conv1", kernel_size = (3,3), strides = 1, filters = 256, padding = "same", activation = "relu", kernel_initializer = initial_weights, kernel_regularizer = regularizer) )
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