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

if __name__ == "__main__":
  image = np.zeros((600,509,3))

  model = Sequential()
  model.add( MaxPooling2D(input_shape = image.shape, pool_size = 2, strides = 2) )
  model.add( MaxPooling2D(pool_size = 2, strides = 2) )
  model.add( MaxPooling2D(pool_size = 2, strides = 2) )
  model.add( MaxPooling2D(pool_size = 2, strides = 2) )
  model.compile(optimizer = SGD(), loss = "binary_crossentropy")

  image = np.expand_dims(image, axis = 0)

  # Light up each pixel
  for y in range(image.shape[1]):
    x = 0
    #for x in range(image.shape[2]):
    image[0,y,x] = [1,1,1]

    feature_map = model.predict(x = image)
    feature_map = np.sum(feature_map, axis = 3)
    where = np.where(feature_map[0,:,:] > 0)
    out_y = where[0][0] if len(where[0]) > 0 else -1
    out_x = where[1][0] if len(where[1]) > 0 else -1
    yy = y // 16
    xx = x // 16

    print("%d,%d -> %d,%d -- %d,%d" % (y, x, out_y, out_x, yy, xx))

    image[0,y,x] = [0,0,0]


