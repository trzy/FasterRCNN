#
# TODO: Add more tests by creating different RoIs and maybe
# a pool size of 3x3
#

from .models.roi_pooling_layer import RoIPoolingLayer
from .models import vgg16

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import Input


if __name__ == "__main__":
  # Some parameters
  num_channels = 512  # input feature map
  pool_size = 7

  
  # Construct test data in PyTorch format
  map_size = (1, 512, 37, 50) # batch,channels,height,width, after VGG16
  feature_map = np.zeros(map_size)
  feature_map[0,0,:,:] = np.arange(map_size[2] * map_size[3]).reshape((map_size[2], -1)) # insert values into channel 0
  indices_and_rois = np.array([
    [0, 267.7272, 315.2921, 408.7873, 371.0944],
  ])

  # Convert to our format: feature map [1,height,width,channels], ROIs: [1,N,4]
  feature_map = np.squeeze(feature_map)
  feature_map = np.transpose(feature_map, [1, 2, 0])
  rois = np.zeros((indices_and_rois.shape[0], 4))
  rois[:,0] = indices_and_rois[:,2] # y1
  rois[:,1] = indices_and_rois[:,1] # x1
  rois[:,2] = indices_and_rois[:,4] # y2
  rois[:,3] = indices_and_rois[:,3] # x2
  rois[:,0:4] = np.round(rois[:,0:4] / 16)
#  rois = vgg16.convert_box_coordinates_from_image_to_output_map_space(box = rois, output_map_shape = feature_map.shape)
  rois[:,2:4] = rois[:,2:4] - rois[:,0:2] + 1
  rois = np.expand_dims(rois, axis=0)
  feature_map = np.expand_dims(feature_map, axis=0)
  

  #TODO; convert from x1,y1,x2,y2->y,x,h,w

  # Build a model to test just the RoI layer using the Keras functional API
  input_map = Input(shape = feature_map.shape[1:])
  input_rois = Input(shape = rois.shape[1:], dtype = tf.int32)
  output_roi_pool = RoIPoolingLayer(pool_size = pool_size)([input_map, input_rois])
  roi_model = Model([input_map, input_rois], output_roi_pool)
  roi_model.summary()
  
  # Run the model
  print(feature_map.shape, rois.shape)
  print(rois)
  x = [ feature_map, rois ]
  y = roi_model.predict(x = x)

  # Print the resulting pool
  print(y.shape)
  print(y[0,0,:,:,0])
