from .intersection_over_union import intersection_over_union
from . import vgg16
from .nms import nms

from collections import defaultdict
import itertools
from math import exp
from math import log
from math import sqrt
import numpy as np
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import models
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

def layers(input_map, l2 = 0):
  assert len(input_map.shape) == 4
  anchors_per_location = 9

  regularizer = tf.keras.regularizers.l2(l2)

  # 3x3 convolution over input map producing 512-d result at each output. The center of each output is an anchor point (k anchors at each point).
  anchors = Conv2D(name = "rpn_conv1", kernel_size = (3,3), strides = 1, filters = 512, padding = "same", activation = "relu", kernel_initializer = "normal", kernel_regularizer = regularizer)(input_map)

  # Classification layer: predicts whether there is an object at the anchor or not. We use a sigmoid function, where > 0.5 is indicates a positive result.
  classifier = Conv2D(name = "rpn_class", kernel_size = (1,1), strides = 1, filters = anchors_per_location, padding = "same", activation = "sigmoid", kernel_initializer = "uniform", kernel_regularizer = regularizer)(anchors)

  # Regress
  regressor = Conv2D(name = "rpn_boxes", kernel_size = (1,1), strides = 1, filters = 4 * anchors_per_location, padding = "same", activation = "linear", kernel_initializer = "zero", kernel_regularizer = regularizer)(anchors)

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
  widths = [ sqrt(areas[i] / x_aspects[j]) for (i, j) in itertools.product(range(3), range(3)) ]
  heights = [x_aspects[j] * sqrt(areas[i] / x_aspects[j]) for (i, j) in itertools.product(range(3), range(3)) ]

  return heights, widths


def compute_all_anchor_boxes(input_image_shape):
  """
  Returns a map of shape (height, width, k*4) where height and width are the
  same as the anchor map and k = 9 different anchor boxes. The anchor boxes are
  laid out as a series of 4 values (center_y, center_x, width, height), all in
  input image space, each following the previous anchor's 4 values.

  Example:

    output[0, 0, 0] = anchor map position (0,0), first anchor center_y
    output[0, 0, 1] = anchor map position (0,0), first anchor center_x
    output[0, 0, 2] = anchor map position (0,0), first anchor height
    output[0, 0, 3] = anchor map position (0,0), first anchor width
    output[0, 0, 4] = anchor map position (0,0), second anchor center_y
    ...

  Also returns a map of shape (height, width, k) indicating valid anchors.
  Anchors that would intersect image boundaries are not valid.
  """
  image_height = input_image_shape[0]
  image_width = input_image_shape[1]

  anchors_per_location = 9  # this is k
  anchor_map_height, anchor_map_width = vgg16.compute_output_map_shape(input_image_shape = input_image_shape)

  # Generate two matrices of same shape as anchor map containing the center coordinate in anchor map space
  anchor_center_x = np.repeat(np.arange(anchor_map_width).reshape((1,anchor_map_width)), repeats = anchor_map_height, axis = 0)
  anchor_center_y = np.repeat(np.arange(anchor_map_height).reshape((anchor_map_height,1)), repeats = anchor_map_width, axis = 1)

  # Convert to input image space
  anchor_center_y, anchor_center_x = vgg16.convert_coordinate_from_output_map_to_image_space(y = anchor_center_y, x = anchor_center_x)

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

  - Map of shape (height, width, k, 8), where height, width, and k refer to
    anchor map shape and number of anchors at each location. The last dimension
    contains:

      0: 0.0 (unused, initialized to 0, for use by caller)
      1: object (1 if this is an object anchor, 0 if either a negative or
         ignored/unused sample)
      2: class (negative if not an object, zero if indeterminate and should be
         ignored, and when positive, equal to one plus the ground truth object
         index)                                                                                 <-- TODO: is this useful?
      3: IoU score with the ground truth box (if this is a positive sample)
      4: ty (if this is a positive sample)
      5: tx (if this is a positive sample)
      6: th (if this is a positive sample)
      7: tw (if this is a positive sample)

  - List of positive anchors (anchors that are classified as "object"), with
    each element consisting of the tuple (y,x,k) indicating the anchor position
    in the previous map.

  - List of all negative anchors.
  """

  assert anchor_boxes.shape[0] == anchor_boxes_valid.shape[0]       # height
  assert anchor_boxes.shape[1] == anchor_boxes_valid.shape[1]       # width
  assert anchor_boxes.shape[2] == anchor_boxes_valid.shape[2] * 4   # k*4
  assert anchor_boxes_valid.shape[2] == 9                           # k=9

  height = anchor_boxes.shape[0]
  width = anchor_boxes.shape[1]
  num_anchors = anchor_boxes_valid.shape[2]
  truth_map = np.zeros((height, width, num_anchors, 8))

  # Compute IoU of each anchor with each box and store the results in a map of
  # shape: (height, width, num_anchors, num_ground_truth_boxes)
  num_ground_truth_boxes = len(ground_truth_object_boxes)
  ious = np.full(shape = (height, width, num_anchors, num_ground_truth_boxes), fill_value = -1.0)
  for y in range(truth_map.shape[0]):
    for x in range(truth_map.shape[1]):
      for k in range(truth_map.shape[2]):

        # Ignore invalid anchors (i.e., at image boundary)
        if not anchor_boxes_valid[y,x,k]:
          continue

        for box_idx in range(num_ground_truth_boxes):
          # Compute anchor box in pixels
          anchor_center_y, anchor_center_x, anchor_height, anchor_width = anchor_boxes[y,x,k*4+0:k*4+4]
          anchor_box_coords = (anchor_center_y - 0.5 * anchor_height, anchor_center_x - 0.5 * anchor_width, anchor_center_y + 0.5 * anchor_height, anchor_center_x + 0.5 * anchor_width)

          # Compute IoU with ground truth box
          box = ground_truth_object_boxes[box_idx]
          object_box_coords = (box.y_min, box.x_min, box.y_max, box.x_max)
          ious[y,x,k,box_idx] = intersection_over_union(box1 = anchor_box_coords, box2 = object_box_coords)

  # Keep track of how many anchors have been assigned to represent each box
  num_anchors_for_box = np.zeros(num_ground_truth_boxes)

  # Associate anchors to ground truth boxes when IoU > 0.7 and background when
  # IoU < 0.3
  for y in range(truth_map.shape[0]):
    for x in range(truth_map.shape[1]):
      for k in range(truth_map.shape[2]):
        if not anchor_boxes_valid[y,x,k]:
          continue
        # Box with highest IoU that exceeds threshold will be associated with
        # this anchor
        best_box_idx = np.argmax(ious[y,x,k,:])
        iou = ious[y,x,k,best_box_idx]
        if iou > 0.7:
          truth_map[y,x,k,1] = 1.0                # this is an object anchor
          truth_map[y,x,k,2] = 1.0 + best_box_idx # object anchor and the box it corresponds to
          num_anchors_for_box[best_box_idx] += 1
        elif iou < 0.3:
          truth_map[y,x,k,1] = 0.0                # this is not an object (background or neutral)
          truth_map[y,x,k,2] = -1.0               # background
        truth_map[y,x,k,3] = iou

  # For each box that still lacks an anchor, construct a list of (iou, (y,x,k))
  # of anchors still available for use (all negative or unassigned anchors)
  anchor_candidates_for_anchorless_box = defaultdict(list)
  for box_idx in range(num_ground_truth_boxes):
    if num_anchors_for_box[box_idx] == 0:
      for y in range(truth_map.shape[0]):
        for x in range(truth_map.shape[1]):
          for k in range(truth_map.shape[2]):
            # Skip invalid anchors
            if not anchor_boxes_valid[y,x,k]:
              continue
            # If this anchor has not been marked as an object, put it in the candidate list
            if truth_map[y,x,k,1] < 1.0:
              iou = ious[y,x,k,box_idx]
              candidate = (iou, (y, x, k))
              anchor_candidates_for_anchorless_box[box_idx].append(candidate)

  # For the unaccounted boxes, pick anchors to assign them. We want to pick the
  # highest IoU anchors to assign to each box. If an anchor has the highest IoU
  # for more than one box (highly unlikely!), it is assigned to the box which
  # has the highest value of the IoU score.
  for box_idx in anchor_candidates_for_anchorless_box:
    candidates = anchor_candidates_for_anchorless_box[box_idx]
    sorted_candidates = sorted(candidates, key = lambda candidate: candidate[0], reverse = True)  # sort descending by IoU
    anchor_candidates_for_anchorless_box[box_idx] = sorted_candidates
  while len(anchor_candidates_for_anchorless_box) > 0:
    # Find the box that currently has the highest-scoring candidate
    box_idx = max(anchor_candidates_for_anchorless_box, key = lambda idx: anchor_candidates_for_anchorless_box[idx][0])

    # Assign best anchor to that box and remove this box from further
    # consideration
    best_candidate = anchor_candidates_for_anchorless_box[box_idx][0] # best candidate is at front of list
    del anchor_candidates_for_anchorless_box[box_idx]
    y, x, k = best_candidate[1][0], best_candidate[1][1], best_candidate[1][2]
    truth_map[y,x,k,1] = 1.0                # object anchor
    truth_map[y,x,k,2] = 1.0 + box_idx      # object anchor and the box it corresponds to
    truth_map[y,x,k,3] = best_candidate[0]  # IoU score

    # Go through all remaining boxes' lists and remove this anchor
    for box_idx in anchor_candidates_for_anchorless_box:
      candidates = anchor_candidates_for_anchorless_box[box_idx]
      candidates = [ candidate for candidate in candidates if candidate[1] != best_candidate[1] ]
      anchor_candidates_for_anchorless_box[box_idx] = candidates

    # Virtually impossible but being pedantic: remove empty lists
    n = len(anchor_candidates_for_anchorless_box)
    anchor_candidates_for_anchorless_box = { box_idx: candidates for box_idx, candidates in anchor_candidates_for_anchorless_box.items() }
    assert n == len(anchor_candidates_for_anchorless_box), "Unexpectedly ran out of anchors to assign to ground truth box"

  # Compute regression parameters of each positive anchor onto ground truth box
  # and, while we're at it, find all the positive and negative anchors
  object_anchors = []
  not_object_anchors = []
  for y in range(truth_map.shape[0]):
    for x in range(truth_map.shape[1]):
      for k in range(truth_map.shape[2]):
        # Compute regression parameters for positive samples only
        if truth_map[y,x,k,1] > 0:
          box_idx = int(truth_map[y,x,k,2] - 1.0)
          anchor_center_y, anchor_center_x, anchor_height, anchor_width = anchor_boxes[y,x,k*4+0:k*4+4]
          anchor_box_coords = (anchor_center_y - 0.5 * anchor_height, anchor_center_x - 0.5 * anchor_width, anchor_center_y + 0.5 * anchor_height, anchor_center_x + 0.5 * anchor_width)
          box = ground_truth_object_boxes[box_idx]
          object_box_coords = (box.y_min, box.x_min, box.y_max, box.x_max)
          center_x = 0.5 * (box.x_min + box.x_max)
          center_y = 0.5 * (box.y_min + box.y_max)
          box_width = box.x_max - box.x_min
          box_height = box.y_max - box.y_min
          ty = (center_y - anchor_center_y) / anchor_height
          tx = (center_x - anchor_center_x) / anchor_width
          th = log(box_height / anchor_height)
          tw = log(box_width / anchor_width)
          truth_map[y,x,k,4:8] = ty, tx, th, tw

        # Store positive and negative samples (but nt neutral ones), in lists
        if truth_map[y,x,k,2] > 0:
          object_anchors.append((y, x, k))
        elif truth_map[y,x,k,2] < 0:
          not_object_anchors.append((y, x, k))

  return truth_map, object_anchors, not_object_anchors

def extract_proposals(y_predicted_class, y_predicted_regression, y_true, anchor_boxes):
  """
  Inputs:

    y_predicted_class: Objectness class predictions map from forward pass of
      RPN model.
    y_predicted_regression: RPN regression predictions.
    y_true: Ground truth anchor map (from compute_anchor_label_assignments())
      containing all anchors for image.
    anchor_boxes: Anchor boxes (compute_all_anchor_boxes()). 

  Returns a map of shape (Nx5) of N proposals from the prediction. Each
  proposal consists of:

    0: y_min
    1: x_min
    2: y_max
    3: x_max
    4: class score (0=background, 1=object)
  """
  y_valid = y_true[:,:,:,:,0] # make sure to filter by valid anchors
  positive_indices = np.argwhere(y_valid * y_predicted_class > 0.5)
  num_proposals = positive_indices.shape[0]
  proposals = np.empty((num_proposals, 5))
  for i in range(proposals.shape[0]):
    _, y_idx, x_idx, k_idx = positive_indices[i]  # first element is sample number, which is always 0: (0, y, x, k)
    box_params = y_predicted_regression[0, y_idx, x_idx, k_idx*4+0 : k_idx*4+4]
    anchor_box = anchor_boxes[y_idx, x_idx, k_idx*4+0 : k_idx*4+4]
    proposals[i,0:4] = convert_parameterized_box_to_points(box_params = box_params, anchor_center_y = anchor_box[0], anchor_center_x = anchor_box[1], anchor_height = anchor_box[2], anchor_width = anchor_box[3])
    proposals[i,4] = y_predicted_class[0, y_idx, x_idx, k_idx]
  proposal_indices = nms(proposals = proposals, iou_threshold = 0.7)
  proposals = proposals[proposal_indices]
  return proposals

def convert_parameterized_box_to_points(box_params, anchor_center_y, anchor_center_x, anchor_height, anchor_width):
  ty, tx, th, tw = box_params
  center_x = anchor_width * tx + anchor_center_x
  center_y = anchor_height * ty + anchor_center_y
  width = exp(tw) * anchor_width
  height = exp(th) * anchor_height
  y_min = center_y - 0.5 * height
  x_min = center_x - 0.5 * width
  y_max = center_y + 0.5 * height
  x_max = center_x + 0.5 * width
  return (y_min, x_min, y_max, x_max)

def clip_box_coordinates_to_map_boundaries(boxes, map_shape):
  """
  Clips an array of image-space boxes provided as an (Nx4) tensor against the
  image boundaries.

  Each box consists of:
  0: y_min
  1: x_min
  2: y_max
  3: x_max
  """
  # First, remove boxes that are entirely out of bounds
  out_of_bounds = (boxes[:,0] >= map_shape[0]) + (boxes[:,2] < 0) + \
                  (boxes[:,1] >= map_shape[1]) + (boxes[:,3] < 0)
  invalid_indices = np.squeeze(np.argwhere(out_of_bounds))  # rows (boxes) that are entirely out of bounds and must be removed
  boxes = np.delete(boxes, invalid_indices, axis = 0)

  # Next, clip to boundaries
  y_max = map_shape[0] - 1
  x_max = map_shape[1] - 1
  boxes = np.maximum(boxes, [ 0, 0, 0, 0 ])                 # clip to x=0 and y=0
  boxes = np.minimum(boxes, [ y_max, x_max, y_max, x_max ]) # clip to maximum dimension

  return boxes

def label_proposals(proposals, ground_truth_object_boxes, num_classes):
  # One hot encoded labels
  num_proposals = proposals.shape[0]
  y_labels = np.zeros((num_proposals, num_classes))

  # IoU threshold for positive examples as in FasterRCNN paper. Note that older
  # models had a minimum threshold (e.g., 0.1), creating a range for negative
  # (background) labels. Presumably proposals that scored even lower than this
  # against all ground truth boxes would have been ignored entirely and removed
  # from the proposal set, which we do not currently support here.
  iou_threshold = 0.5  

  for i in range(num_proposals):
    best_iou = 0
    best_class_idx = 0  # background 

    for box in ground_truth_object_boxes:
      proposal_box_coords = proposals[i,0:4]
      object_box_coords = np.array([box.y_min, box.x_min, box.y_max, box.x_max])
      iou = intersection_over_union(box1 = proposal_box_coords, box2 = object_box_coords)
      if iou > best_iou:
        best_iou = iou
        best_class_idx = box.class_index

    # Create one-hot encoded label for this proposal
    y_labels[i,best_class_idx] = 1.0

  return y_labels
