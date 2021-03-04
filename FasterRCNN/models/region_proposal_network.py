import itertools
from math import sqrt
import numpy as np
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import models
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

def layers(input_map):
  assert len(input_map.shape) == 4
  anchors_per_location = 9
  
  # We can infer the total number of anchors from the input map size
  height = input_map.shape[1]
  width = input_map.shape[2]
  num_anchors = anchors_per_location * height * width

  # 3x3 convolution over input map producing 512-d result at each output. The center of each output is an anchor point (k anchors at each point).
  anchors = Conv2D(name = "rpn_conv1", kernel_size = (3,3), strides = 1, filters = 512, padding = "same", activation = "relu", kernel_initializer = "normal")(input_map)

  # Classification layer: predicts whether there is an object at the anchor or not. We use a sigmoid function, where > 0.5 is indicates a positive result.
  classifier = Conv2D(name = "rpn_class", kernel_size = (1,1), strides = 1, filters = anchors_per_location, padding = "same", activation = "sigmoid", kernel_initializer = "uniform")(anchors)
  
  # Regress
  regressor = Conv2D(name = "rpn_boxes", kernel_size = (1,1), strides = 1, filters = 4 * anchors_per_location, padding = "same", activation = "linear", kernel_initializer = "zero")(anchors)

  return [ classifier, regressor ]

def convert_anchor_coordinate_from_rpn_layer_to_image_space(y, x, image_input_map, anchor_map):
  """
  Returns (y, x) converted from the coordinate space of an anchor map (the
  output of the RPN model) to the original input image space. This gives the
  anchor center coordinates in the original image.
  """
  image_height = image_input_map.shape[1]
  image_width = image_input_map.shape[2]
  downstream_height = anchor_map.shape[1]
  downstream_width = anchor_map.shape[2]
  y_scale = image_height / downstream_height      # will be pretty close to 16 for VGG
  x_scale = image_width / downstream_width
  return (y + 0.5) * y_scale, (x + 0.5) * x_scale # add 0.5 to move into the center of the cell

def compute_all_anchor_boxes(image_input_map, anchor_map):
  """
  Returns a map of shape (height, width, k*4) where height and width are the
  same as the anchor map and k = 9 different anchor boxes. The anchor boxes are
  laid out as a series of 4 values (center_y, center_x, width, height), all in
  input image space, each following the previous anchor's 4 values.

  Example:

    output[0, 0, 0] = anchor map position (0,0), first anchor center_y
    output[0, 0, 1] = anchor map position (0,0), first anchor center_x
    output[0, 0, 2] = anchor map position (0,0), first anchor width
    output[0, 0, 3] = anchor map position (0,0), first anchor height
    output[0, 0, 4] = anchor map position (0,0), second anchor center_y
    ...
  """
  anchors_per_location = 9  # this is k
  anchor_map_height = anchor_map.shape[1]
  anchor_map_width = anchor_map.shape[2]

  # Generate two matrices of same shape as anchor map containing the center coordinate in anchor map space
  anchor_center_x = np.repeat(np.arange(anchor_map_width).reshape((1,anchor_map_width)), repeats = anchor_map_height, axis = 0)
  anchor_center_y = np.repeat(np.arange(anchor_map_height).reshape((anchor_map_height,1)), repeats = anchor_map_width, axis = 1)

  # Convert to input image space
  anchor_center_y, anchor_center_x = convert_anchor_coordinate_from_rpn_layer_to_image_space(y = anchor_center_y, x = anchor_center_x, image_input_map = image_input_map, anchor_map = anchor_map)

  #
  # Anchor scales and aspect ratios.
  #
  # x * y = area          x * (x_aspect * x) = x_aspect * x^2 = area
  # x_aspect * x = y  ->  x = sqrt(area / x_aspect)
  #                       y = x_aspect * sqrt(area / x_aspect)
  #
  areas = [ 128, 256, 512 ]       # pixels^2
  x_aspects = [ 1.0, 0.5, 2.0 ]   # x:1 ratio

  # Generate all 9 combinations of area and aspect ratio
  widths = [ int(sqrt(areas[i] / x_aspects[j])) for (i, j) in itertools.product(range(3), range(3)) ]
  heights = [ int(x_aspects[j] * sqrt(areas[i] / x_aspects[j])) for (i, j) in itertools.product(range(3), range(3)) ]

  # Create (height, width, k*4) matrix where the last axis is a repeating series of (center_y, center_x, height, width)
  matrices = []
  for i in range(anchors_per_location):
    width_matrix = np.ones((anchor_map_height, anchor_map_width)) * widths[i]
    height_matrix = np.ones((anchor_map_height, anchor_map_width)) * heights[i]
    matrices += [ anchor_center_y, anchor_center_x, height_matrix, width_matrix ]
  anchor_boxes = np.stack(matrices, axis = 2) # stack all k*4 values along third dimension

  return anchor_boxes