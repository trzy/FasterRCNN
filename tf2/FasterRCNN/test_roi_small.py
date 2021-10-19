#
# TODO: Add more tests by creating different RoIs and maybe
# a pool size of 3x3
#

from .models.roi_pooling_layer import RoIPoolingLayer
from .models import vgg16
from .models.region_proposal_network import _compute_anchor_sizes

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import Input


def print_results(y):
  # Print for each sample
  for i in range(y.shape[0]):
    print("Sample %d" % i)
    for roi_num in range(y.shape[1]):
      print("  ROI %d" % roi_num)
      for channel_num in range(y.shape[4]):
        print("    Channel %d" % channel_num)
        for yy in range(y.shape[2]):
          row_values = [ ("%1.2f" % y[i,roi_num,yy,xx,channel_num]) for xx in range(y.shape[3]) ]
          print("      %s" % row_values)

if __name__ == "__main__":
  print(_compute_anchor_sizes())
  exit()
  # Some parameters
  num_channels = 1  # input feature map
  pool_size = 2

  
  # Construct test data in PyTorch format
  map_size = (1, 1, 9, 8) # batch,channels,height,width, after VGG16
  channel_0 = np.array([
    [ 10,  2,  3,  4,  5,  6,  7,  8 ],
    [ 2,  3,  4,  5,  6,  7,  8,  9 ],
    [ 3,  4,  5,  6,  7,  8,  9,  1 ],
    [ 4,  5,  6,  7,  8,  9,  1,  2 ],
    [ 5,  6,  7,  8,  9,  1,  2,  3 ],
    [ 6,  7,  8,  9,  1,  2,  3,  4 ],
    [ 7,  8,  9,  1,  2,  3,  4,  5 ],
    [ 8, 10,  1,  2,  3,  4,  11, 6 ],
    [ 9,  1,  2,  3,  4,  5,  6,  7 ]
  ], dtype = np.float)
  feature_map = np.zeros(map_size)
  feature_map[0,0,:,:] = channel_0 
  indices_and_rois = np.array([ # these must be given in actual pixel coordinates, 16x scale, (x1,y1,x2,y2)
    [ 0, 32, 0, 55, 7],
    #[0, 8, 0, 24, 8],
  ])

  # Convert to our format: feature map [1,height,width,channels], ROIs: [1,N,4]
  feature_map = feature_map.reshape(feature_map.shape[1:])  # squeeze batch dim (which should be just 1 element)
  feature_map = np.transpose(feature_map, [1, 2, 0])
  rois = np.zeros((indices_and_rois.shape[0], 4))
  rois[:,0] = indices_and_rois[:,2] # y1
  rois[:,1] = indices_and_rois[:,1] # x1
  rois[:,2] = indices_and_rois[:,4] # y2
  rois[:,3] = indices_and_rois[:,3] # x2
  print("ROIs (y1,x1,y2,x2):")
  print(rois)
  rois[:,0:4] = np.floor((rois[:,0:4] - 8) / 16) + 1
  #rois[:,0:4] = np.ceil(rois[:,0:4] / 16)
  print("ROIs feature map space, rounded:")
  print(rois)
#  rois = vgg16.convert_box_coordinates_from_image_to_output_map_space(box = rois, output_map_shape = feature_map.shape)
  rois[:,2:4] = rois[:,2:4] - rois[:,0:2] + 1 # (y1,x1,h,w)
  print("ROIs feature map space, (y1,x1,h,w):")
  print(rois)
  rois = np.expand_dims(rois, axis=0)
  feature_map = np.expand_dims(feature_map, axis=0)
  


  # Build a model to test just the RoI layer using the Keras functional API
  input_map = Input(shape = feature_map.shape[1:])
  input_rois = Input(shape = rois.shape[1:], dtype = tf.int32)
  output_roi_pool = RoIPoolingLayer(pool_size = pool_size)([input_map, input_rois])
  roi_model = Model([input_map, input_rois], output_roi_pool)
  roi_model.summary()
  
  # Run the model
  print(feature_map.shape, rois.shape)
  print("ROIs:")
  print(rois)
  x = [ feature_map, rois ]
  y = roi_model.predict(x = x)

  # Print the resulting pool
  print(y.shape)
  print(y[0,0,:,:,0])
  print_results(y)
