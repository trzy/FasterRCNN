#TODO: assert anchor map shape is indeed image/16
from .intersection_over_union import intersection_over_union

import itertools
from math import sqrt
from math import log
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

def _compute_anchor_sizes():
  #
  # Anchor scales and aspect ratios.
  #
  # x * y = area          x * (x_aspect * x) = x_aspect * x^2 = area
  # x_aspect * x = y  ->  x = sqrt(area / x_aspect)
  #                       y = x_aspect * sqrt(area / x_aspect)
  #
  areas = [ 128*128, 256*256, 512*512 ]   # pixels
  x_aspects = [ 1.0, 0.5, 2.0 ]   # x:1 ratio

  # Generate all 9 combinations of area and aspect ratio
  widths = [ int(sqrt(areas[i] / x_aspects[j])) for (i, j) in itertools.product(range(3), range(3)) ]
  heights = [ int(x_aspects[j] * sqrt(areas[i] / x_aspects[j])) for (i, j) in itertools.product(range(3), range(3)) ]

  return heights, widths

def compute_anchor_map_shape(input_image_shape):
  """
  Returns the 2D shape of the RPN output map (height, width), which will be
  1/16th of the input image for VGG16. 
  """
  return (input_image_shape[0] // 16, input_image_shape[1] // 16)

def convert_anchor_coordinate_from_rpn_layer_to_image_space(y, x):
  """
  Returns (y, x) converted from the coordinate space of an anchor map (the
  output of the RPN model) to the original input image space. This gives the
  anchor center coordinates in the original image.
  """
  y_scale = 16
  x_scale = 16
  return (y + 0.5) * y_scale, (x + 0.5) * x_scale # add 0.5 to move into the center of the cell

def compute_all_anchor_boxes(input_image_shape):
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
  image_height = input_image_shape[0]
  image_width = input_image_shape[1]

  anchors_per_location = 9  # this is k
  anchor_map_height, anchor_map_width = compute_anchor_map_shape(input_image_shape = input_image_shape)
  
  # Generate two matrices of same shape as anchor map containing the center coordinate in anchor map space
  anchor_center_x = np.repeat(np.arange(anchor_map_width).reshape((1,anchor_map_width)), repeats = anchor_map_height, axis = 0)
  anchor_center_y = np.repeat(np.arange(anchor_map_height).reshape((anchor_map_height,1)), repeats = anchor_map_width, axis = 1)

  # Convert to input image space
  anchor_center_y, anchor_center_x = convert_anchor_coordinate_from_rpn_layer_to_image_space(y = anchor_center_y, x = anchor_center_x)

  # All possible anchor sizes
  heights, widths = _compute_anchor_sizes()

  #
  # Create the anchor boxes matrix: (height, width, k*4) where the last axis is
  # a repeating series of (center_y, center_x, height, width).
  #
  # Also construct a corresponding matrix indicating box validity:
  # (height, width, k), where each element is a bool indicating whether or not
  # the anchor box is valid (within the image boundaries).
  #
  box_matrices = []
  valid_matrices = []
  for i in range(anchors_per_location):
    width_matrix = np.ones((anchor_map_height, anchor_map_width)) * widths[i]
    height_matrix = np.ones((anchor_map_height, anchor_map_width)) * heights[i]
    box_matrices += [ anchor_center_y, anchor_center_x, height_matrix, width_matrix ]

    # Construct a bool matrix indicating whether the anchor is valid by testing
    # it against image boundaries. Note that multiplication is equivalent to an
    # AND function.
    in_bounds = np.where(anchor_center_y - 0.5 * heights[i] >= 0, True, False) * np.where(anchor_center_x - 0.5 * widths[i] >= 0, True, False) * np.where(anchor_center_y + 0.5 * heights[i] < image_height, True, False) * np.where(anchor_center_x + 0.5 * widths[i] < image_width, True, False)
    valid_matrices.append(in_bounds)

  anchor_boxes = np.stack(box_matrices, axis = 2)         # stack all k*4 values along third dimension
  anchor_boxes_valid = np.stack(valid_matrices, axis = 2) # k values stacked along third dimension

  return anchor_boxes, anchor_boxes_valid

def compute_anchor_label_assignments(ground_truth_object_boxes, anchor_boxes, anchor_boxes_valid):
  """
  Returns:

  - Map of shape (height, width, k, 6), where height, width, and k refer to
    anchor map shape and number of anchors at each location. The last dimension
    contains:

      0: class (negative if not an object, zero if indeterminate and should be
         ignored, and when positive, equal to one plus the ground truth object
         index)
      1: IoU score with the ground truth box (if this is a positive sample)
      2: ty (if this is a positive sample)
      3: tx (if this is a positive sample)
      4: th (if this is a positive sample)
      5: tw (if this is a positive sample)

  - List of positive anchors (anchors that are classified as "object"), with
    each element consisting of the tuple (y,x,k) indicating the anchor position
    in the previous map.

  - List of all negative anchors.
  """

  assert anchor_boxes.shape[0] == anchor_boxes_valid.shape[0]       # height
  assert anchor_boxes.shape[1] == anchor_boxes_valid.shape[1]       # width
  assert anchor_boxes.shape[2] == anchor_boxes_valid.shape[2] * 4   # k*4
  assert anchor_boxes_valid.shape[2] == 9                           # k=9

  # Temporary arrays whose primary index is the ground truth box number [0,N)
  num_boxes = len(ground_truth_object_boxes)
  anchor_assigned_for_box = np.zeros(num_boxes, dtype = np.bool)    # true if at least one anchor meeting the objectness threshold was found for each given ground truth box
  best_anchor_for_box = np.zeros((num_boxes, 8))                    # each entry is: (iou, anchor_y_idx, anchor_x_idx, anchor_k_idx, ty, tx, th, tw)
  
  #
  # Output array corresponding to anchor map, where fourth dimension consists
  # of: (class, iou, ty, tx, th, tw). The "class" is:
  #
  #   - Negative: A negative (not object) anchor.
  #   - Positive: A positive (object) anchor, where the ground truth box index
  #     is this value minus one.
  #   - Zero: An anchor that should not be used as a positive nor negative
  #     example.
  #
  anchors_height = anchor_boxes.shape[0]
  anchors_width = anchor_boxes.shape[1]
  anchors_sizes = anchor_boxes_valid.shape[2]
  regression_map = np.zeros((anchors_height, anchors_width, anchors_sizes, 6))
  
  # First pass over all anchors and ground truth boxes: determine which anchors
  # can be definitively classified as "object" or "not object". It is possible
  # that in this pass, some ground truth boxes will not be associated with an
  # anchor because none will meet the strict "object" IoU threshold.
  for y in range(anchors_height):
    for x in range(anchors_width):
      for k in range(anchors_sizes):

        if not anchor_boxes_valid[y,x,k]:
          continue  # ignore anchors that aren't even valid

        anchor_center_y, anchor_center_x, anchor_height, anchor_width = anchor_boxes[y,x,k*4+0:k*4+4]
        anchor_box_coords = (anchor_center_y - 0.5 * anchor_height, anchor_center_x - 0.5 * anchor_width, anchor_center_y + 0.5 * anchor_height, anchor_center_x + 0.5 * anchor_width)

        #
        # Test against every ground truth box: any anchor that meets an IoU
        # threshold will be labeled as "object" (and regressed box values
        # generated to match the bounding box), those below a second IoU
        # threshold against all ground truth boxes will be labeled as "not
        # object". Anything in between will be unused.
        #
        # Multiple anchors can sufficiently intersect with a single ground
        # truth box. If an anchor intersects more than one ground truth box
        # (highly unlikely), it will simply be assigned to the one with which
        # it had the highest overlap.
        #
        
        num_boxes_positive = 0
        num_boxes_negative = 0
        
        for box_idx in range(num_boxes):
          # Compute IoU between anchor and current ground truth box
          box = ground_truth_object_boxes[box_idx]
          object_box_coords = (box.y_min, box.x_min, box.y_max, box.x_max)
          iou = intersection_over_union(box1 = anchor_box_coords, box2 = object_box_coords)

          # Compute regression of anchor to fit ground truth box
          center_x = 0.5 * (box.x_min + box.x_max)
          center_y = 0.5 * (box.y_min + box.y_max)
          width = box.x_max - box.x_min
          height = box.y_max - box.y_min
          ty = (center_y - anchor_center_y) / anchor_height
          tx = (center_x - anchor_center_x) / anchor_width
          th = log(height / anchor_height)
          tw = log(width / anchor_width)

          # Keep track of the best anchor found for every ground truth box
          if iou > best_anchor_for_box[box_idx,0]:
            best_anchor_for_box[box_idx,0] = iou
            best_anchor_for_box[box_idx,1] = y
            best_anchor_for_box[box_idx,2] = x
            best_anchor_for_box[box_idx,3] = k
            best_anchor_for_box[box_idx,4] = ty
            best_anchor_for_box[box_idx,5] = tx
            best_anchor_for_box[box_idx,6] = th
            best_anchor_for_box[box_idx,7] = tw

          # Label each anchor. We have to be careful not to overwrite a better
          # result belonging to a different box
          if iou > 0.7 and iou > regression_map[y,x,k,1]:
            # This IoU meets our threshold for "object" and is better than any
            # other IoU found for this anchor
            regression_map[y,x,k,0] = box_idx + 1.0 # positive example: box number as [1,N]
            regression_map[y,x,k,1] = iou
            regression_map[y,x,k,2] = ty
            regression_map[y,x,k,3] = tx
            regression_map[y,x,k,4] = th
            regression_map[y,x,k,5] = tw
            anchor_assigned_for_box[box_idx] = True
          elif iou < 0.3 and regression_map[y,x,k,1] <= 0:
            # Definitely not an object and we do not have a positive anchor
            # result at this slot for any other ground truth box
            regression_map[y,x,k,0] = -1.0          # negative value indicates "not object"
  
  # For any ground truth box that did not have an anchor assigned, use the
  # highest-scoring anchor found. This means we may potentially reassign
  # that anchor from a higher-scoring ground truth box -- but do we have a
  # choice? I may be interpreting the Faster R-CNN paper incorrectly here.
  for box_idx in range(len(ground_truth_object_boxes)):
    if not anchor_assigned_for_box[box_idx]:
      # Get the best anchor found
      y = int(best_anchor_for_box[box_idx,1])
      x = int(best_anchor_for_box[box_idx,2])
      k = int(best_anchor_for_box[box_idx,3])

      # Assign this box to it, overwriting previous assignment
      regression_map[y,x,k,0] = box_idx + 1.0
      regression_map[y,x,k,1] = best_anchor_for_box[box_idx,0]
      regression_map[y,x,k,2:6] = best_anchor_for_box[box_idx,4:8]

  # Second pass over all anchors: make lists of all positive and negative
  # examples (for easy random indexing later)
  object_anchors = []
  not_object_anchors = []
  for y in range(anchors_height):
    for x in range(anchors_width):
      for k in range(anchors_sizes):
        if regression_map[y,x,k,0] > 0:
          object_anchors.append((y, x, k))
        elif regression_map[y,x,k,0] < 0:
          not_object_anchors.append((y, x, k))

  return regression_map, object_anchors, not_object_anchors


