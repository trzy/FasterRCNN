#
# TODO: Add more tests by creating different RoIs and maybe
# a pool size of 3x3
#

from ..models.roi_pooling_layer import RoIPoolingLayer

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
  # Some parameters
  num_channels = 2  # input feature map
  pool_size = 2
  num_rois = 4

  # Build a model to test just the RoI layer using the Keras functional API
  input_map = Input(shape = (9,8,pool_size))                  # input map size
  input_rois = Input(shape = (num_rois,4), dtype = tf.int32)  # N RoIs, each of length 4 (y,x,h,w)
  output_roi_pool = RoIPoolingLayer(pool_size = pool_size, num_rois = num_rois)([input_map, input_rois])
  roi_model = Model([input_map, input_rois], output_roi_pool)
  roi_model.summary()


  # Create 2-channel input map
  channel_0 = np.array([
    [ 1,  2,  3,  4,  5,  6,  7,  8 ],
    [ 2,  3,  4,  5,  6,  7,  8,  9 ],
    [ 3,  4,  5,  6,  7,  8,  9,  1 ],
    [ 4,  5,  6,  7,  8,  9,  1,  2 ],
    [ 5,  6,  7,  8,  9,  1,  2,  3 ],
    [ 6,  7,  8,  9,  1,  2,  3,  4 ],
    [ 7,  8,  9,  1,  2,  3,  4,  5 ],
    [ 8, 10,  1,  2,  3,  4,  11, 6 ],
    [ 9,  1,  2,  3,  4,  5,  6,  7 ]
  ], dtype = np.float)

  channel_1 = np.array([
    [ 0.88, 0.44, 0.14, 0.16, 0.37, 0.77, 0.96, 0.27 ],
    [ 0.19, 0.45, 0.57, 0.16, 0.63, 0.29, 0.71, 0.70 ],
    [ 0.66, 0.26, 0.82, 0.64, 0.54, 0.73, 0.59, 0.26 ],
    [ 0.85, 0.34, 0.76, 0.84, 0.29, 0.75, 0.62, 0.25 ],
    [ 0.32, 0.74, 0.21, 0.39, 0.34, 0.03, 0.33, 0.48 ],
    [ 0.20, 0.14, 0.16, 0.13, 0.73, 0.65, 0.96, 0.32 ],
    [ 0.19, 0.69, 0.09, 0.86, 0.88, 0.07, 0.01, 0.48 ],
    [ 0.83, 0.24, 0.97, 0.04, 0.24, 0.35, 0.50, 0.91 ],
    [ 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000 ]
  ], dtype = np.float)

  assert num_channels == 2
  test_input_map = np.stack([channel_0, channel_1], axis = 2).reshape((9,8,num_channels))

  # Create 4 RoIs
  test_input_rois = np.array([
    [ 3, 0, 5, 7 ],
    [ 0, 0, 1, 1 ],
    [ 3, 0, 5, 7 ],
    [ 3, 0, 5, 7 ]
  ], dtype = np.int)


  # Create a 2-batch input, where second input map has all values multiplied by 2
  x_maps = np.array([ test_input_map, 2*test_input_map ])
  x_rois = np.array([ test_input_rois, test_input_rois ])

  # Run the model
  x = [ x_maps, x_rois ]
  y = roi_model.predict(x = x)

  # Print the resulting pool
  print(y.shape)
  print_results(y)

  # Expected results
  expected_pools = [
    # Sample 0
    [ # RoI 0, Channel 0
      [ 7,  9 ],
      [ 10, 11 ]
    ],

    [ # RoI 0, Channel 1
      [ 0.85, 0.84 ],
      [ 0.97, 0.96 ]
    ],

    [ # RoI 1, Channel 0
      [ 1, 1 ],
      [ 1, 1 ]
    ],

    [ # RoI 1, Channel 1
      [ 0.88, 0.88 ],
      [ 0.88, 0.88 ]
    ],

    [ # RoI 2, Channel 0
      [ 7,  9 ],
      [ 10, 11 ]
    ],

    [ # RoI 2, Channel 1
      [ 0.85, 0.84 ],
      [ 0.97, 0.96 ]
    ],

    [ # RoI 3, Channel 0
      [ 7,  9 ],
      [ 10, 11 ]
    ],

    [ # RoI 3, Channel 1
      [ 0.85, 0.84 ],
      [ 0.97, 0.96 ]
    ],

    # Sample 1
    [ # RoI 0, Channel 0
      [ 14, 18 ],
      [ 20, 22 ]
    ],

    [ # RoI 0, Channel 1
      [ 1.70, 1.68 ],
      [ 1.94, 1.92 ]
    ],

    [ # RoI 1, Channel 0
      [ 2, 2 ],
      [ 2, 2 ]
    ],

    [ # RoI 1, Channel 1
      [ 1.76, 1.76 ],
      [ 1.76, 1.76 ]
    ],

    [ # RoI 2, Channel 0
      [ 14, 18 ],
      [ 20, 22 ]
    ],

    [ # RoI 2, Channel 1
      [ 1.70, 1.68 ],
      [ 1.94, 1.92 ]
    ],

    [ # RoI 3, Channel 0
      [ 14, 18 ],
      [ 20, 22 ]
    ],

    [ # RoI 3, Channel 1
      [ 1.70, 1.68 ],
      [ 1.94, 1.92 ]
    ]
  ]

  # Turn it into a tensor of appropriate shape. For convenience, we interleaved the two
  # channels and the result will have shape (num_channels*num_samples*num_rois, pool_size, pool_size)
  # but we must convert to: (num_samples, num_rois, pool_size, pool_size, num_channels).
  channel_0 = expected_pools[0::2]
  channel_1 = expected_pools[1::2]
  expected_pools = np.stack([channel_0, channel_1], axis = 3) # (num_samples*num_rois, pool_size, pool_size, num_channels)
  expected_pools = expected_pools.reshape((2,num_rois,pool_size,pool_size,num_channels))

  # Check result
  assert np.sum(expected_pools - y) < 1e-7, "** Test FAILED **"
  print(np.sum(expected_pools - y))
  print("** Test PASSED **")