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
import time

def layers(input_map, l2 = 0):
  assert len(input_map.shape) == 4
  anchors_per_location = 9

  regularizer = tf.keras.regularizers.l2(l2)

  # 3x3 convolution over input map producing 512-d result at each output. The center of each output is an anchor point (k anchors at each point).
  anchors = Conv2D(name = "rpn_conv1", kernel_size = (3,3), strides = 1, filters = 512, padding = "same", activation = "relu", kernel_initializer = "normal", kernel_regularizer = regularizer)(input_map)

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

def compute_ground_truth_map(ground_truth_object_boxes, anchor_boxes, anchor_boxes_valid):
  """
  Returns:

  - Ground truth map of shape (height, width, k, 8), where height, width, and k
    refer to anchor map shape and number of anchors at each location. The last
    dimension contains:

      0: valid -- 0 if anchor is not valid (because it crosses image bounds),
         otherwise 1 if it can be used to generate object proposals. This is
         just a copy of anchor_boxes_valid passed into this function.
      1: object (1 if this is an object anchor, 0 if either a negative or
         ignored/unused sample). Set to 0 for invalid anchors (those that
         overlap image boundaries).
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
  
  # Initialize the validity field
  truth_map[:,:,:,0] = anchor_boxes_valid

  # Compute IoU of each anchor with each box and store the results in a map of
  # shape: (height, width, num_anchors, num_ground_truth_boxes)
  __t0 = time.perf_counter()
  num_ground_truth_boxes = len(ground_truth_object_boxes)
  ious = np.full(shape = (height, width, num_anchors, num_ground_truth_boxes), fill_value = -1.0)
   
  anchors_center_y = anchor_boxes[:, :, 0::4]           # each map has shape (height,width,k)
  anchors_center_x = anchor_boxes[:, :, 1::4]
  anchors_height = anchor_boxes[:, :, 2::4]
  anchors_width = anchor_boxes[:, :, 3::4]
  anchors_y1 = anchors_center_y - 0.5 * anchors_height
  anchors_x1 = anchors_center_x - 0.5 * anchors_width
  anchors_y2 = anchors_center_y + 0.5 * anchors_height
  anchors_x2 = anchors_center_x + 0.5 * anchors_width
  anchors_area = anchors_height * anchors_width
  
  for box_idx in range(num_ground_truth_boxes):
    box = ground_truth_object_boxes[box_idx]
    box_y1, box_x1, box_y2, box_x2 = box.corners
    box_area = (box_y2 - box_y1 + 1) * (box_x2 - box_x1 + 1) 
    
    # Compute IoU of this box against every anchor
    y1 = np.maximum(box_y1, anchors_y1)
    x1 = np.maximum(box_x1, anchors_x1)
    y2 = np.minimum(box_y2, anchors_y2)
    x2 = np.minimum(box_x2, anchors_x2)
    heights = np.maximum(y2 - y1 + 1, 0.0)
    widths = np.maximum(x2 - x1 + 1, 0.0)
    intersections = heights * widths
    unions = box_area + anchors_area - intersections
    ious[:,:,:,box_idx] = intersections / unions

  # No IoU for invalid anchors positions
  for y in range(truth_map.shape[0]):
    for x in range(truth_map.shape[1]):
      for k in range(truth_map.shape[2]):
        if not anchor_boxes_valid[y,x,k]:
          ious[y,x,k,:] = -1.0
  __compute_iou_time = time.perf_counter() - __t0

  # Keep track of how many anchors have been assigned to represent each box
  num_anchors_for_box = np.zeros(num_ground_truth_boxes)

  # Associate anchors to ground truth boxes when IoU > 0.7 and background when
  # IoU < 0.3
  __t0 = time.perf_counter()
  best_box_idxs = np.argmax(ious, axis = 3)   # shape (height,width,k) of best ground truth box index (highest IoU) for each anchor position
  best_ious = np.max(ious, axis = 3)          # shape (height,width,k), containing corresponding best IoU 
  for y in range(truth_map.shape[0]):
    for x in range(truth_map.shape[1]):
      for k in range(truth_map.shape[2]):
        if not anchor_boxes_valid[y,x,k]:
          continue
        # Box with highest IoU that exceeds threshold will be associated with
        # this anchor
        best_box_idx = best_box_idxs[y,x,k]
        iou = best_ious[y,x,k]
        if iou > 0.7:
          truth_map[y,x,k,1] = 1.0                # this is an object anchor
          truth_map[y,x,k,2] = 1.0 + best_box_idx # object anchor and the box it corresponds to
          num_anchors_for_box[best_box_idx] += 1
        elif iou < 0.3:
          truth_map[y,x,k,1] = 0.0                # this is not an object (background or neutral)
          truth_map[y,x,k,2] = -1.0               # background
        truth_map[y,x,k,3] = iou
  __associate_boxes_time = time.perf_counter() - __t0

  # For each box that still lacks an anchor, construct a list of (iou, (y,x,k))
  # of anchors still available for use (all negative or unassigned anchors)
  __t0 = time.perf_counter()
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
  __anchor_candidates_time = time.perf_counter() - __t0

  # For the unaccounted boxes, pick anchors to assign them. We want to pick the
  # highest IoU anchors to assign to each box. If an anchor has the highest IoU
  # for more than one box (highly unlikely!), it is assigned to the box which
  # has the highest value of the IoU score.
  __t0 = time.perf_counter()
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
  __unaccounted_pairing_time = time.perf_counter() - __t0

  # Compute regression parameters of each positive anchor onto ground truth box
  __t0 = time.perf_counter()
  for y in range(truth_map.shape[0]):
    for x in range(truth_map.shape[1]):
      for k in range(truth_map.shape[2]):
        # Compute regression parameters for positive samples only
        if truth_map[y,x,k,1] > 0:
          box_idx = int(truth_map[y,x,k,2] - 1.0)
          anchor_center_y, anchor_center_x, anchor_height, anchor_width = anchor_boxes[y,x,k*4+0:k*4+4]
          anchor_box_coords = (anchor_center_y - 0.5 * anchor_height, anchor_center_x - 0.5 * anchor_width, anchor_center_y + 0.5 * anchor_height, anchor_center_x + 0.5 * anchor_width)
          box = ground_truth_object_boxes[box_idx]
          center_x = 0.5 * (box.corners[1] + box.corners[3])  # 0.5 * (x_min + x_max)
          center_y = 0.5 * (box.corners[0] + box.corners[2])  # 0.5 * (y_min + y_max)
          box_width = box.corners[3] - box.corners[1] + 1     # x_max - x_min + 1
          box_height = box.corners[2] - box.corners[0] + 1    # y_max - y_min + 1
          ty = (center_y - anchor_center_y) / anchor_height
          tx = (center_x - anchor_center_x) / anchor_width
          th = log(box_height / anchor_height)
          tw = log(box_width / anchor_width)
          truth_map[y,x,k,4:8] = ty, tx, th, tw
 
  # Store positive and negative samples (but not neutral ones) in lists
  truth_map_coords = np.transpose(np.mgrid[0:height,0:width,0:num_anchors], (1,2,3,0))  # shape (height,width,k,3): every index (y,x,k,:) returns its own coordinate (y,x,k)
  object_anchors = truth_map_coords[np.where(truth_map[:,:,:,2] > 0)]                   # shape (N,3), where each row is the coordinate (y,x,k) of a positive sample
  not_object_anchors = truth_map_coords[np.where(truth_map[:,:,:,2] < 0)]               # shape (N,3), where each row is the coordinate (y,x,k) of a negative sample

  __regression_param_time = time.perf_counter() - __t0

  #print("---")
  #print("Compute IoU Time        :", __compute_iou_time)
  #print("Associate Boxes Time    :", __associate_boxes_time)
  #print("Anchor Candidates Time  :", __anchor_candidates_time)
  #print("Unaccounted Pairing Time:", __unaccounted_pairing_time)
  #print("Regression Param Time   :", __regression_param_time)

  return truth_map, object_anchors, not_object_anchors

def extract_proposals(y_predicted_class, y_predicted_regression, input_image_shape, anchor_boxes, anchor_boxes_valid, max_proposals = 0):
  """
  Extracts object proposals from the outputs of the region proposal network.

  Parameters
  ----------
    y_predicted_class : np.ndarray
      Object predictions map from forward pass of RPN model with shape
      shape (1,h,w,k), where h and w are the height and width, respectively, in
      terms of anchors (i.e., the RPN output map dimensions) and k is the
      number of anchors (i.e., 9). Batch size must be 1. A score of 0 indicates
      background and 1 is an object.
    y_predicted_regression : np.ndarray
      Predicted bounding box regression parameters for each possible anchor.
      Has shape (1,h,w,k*4), where the last dimension holds the values (ty, tx,
      tw, th).
    input_image_shape : (int, int, int)
      Shape of input image in pixels, (height, width, channels). Used to clip
      proposals against image boundaries.
    anchor_boxes : np.ndarray
      Map of shape (h,w,k*4) describing all possible anchors, where the last
      dimension is a series of 4-tuples of (center_y,center_x,height,width), in
      input image pixel units, of each of the k anchors.
    anchor_boxes_valid : np.ndarray
      Mask of valid anchors available for use during inference or training.
      Has shape (h,w,k). A value of 1.0 is valid and 0.0 is invalid.
    max_proposals : int
      Maximum number of proposals to extract. The top proposals, sorted by
      descending object scores, are selected. If <= 0, all proposals are used.

  Returns
  -------
  np.ndarray
    A map of shape (N,5) of N object proposals from the prediction. Each 
    proposal consists of box coordinates in input image pixel space and
    the objectness score:

      0: y_min
      1: x_min
      2: y_max
      3: x_max
      4: score
  """
  #
  # Find all valid proposals (object score > 0.5 and at a valid anchor) and
  # construct a proposal map of shape (N,5), containing the proposal box 
  # coordinates and objectness score.
  #
  # Note that because we assume a batch size of 1, it is safe to multiply
  # anchor_boxes_valid, with shape (y,x,k), with a map of shape (1,y,x,k),
  # without needing to expand its dimensions.
  #
  positive_indices = np.argwhere(anchor_boxes_valid * y_predicted_class > 0.5)  # positive indices at valid anchor locations
  num_proposals = positive_indices.shape[0]
  proposals = np.empty((num_proposals, 5))
  for i in range(proposals.shape[0]):
    _, y_idx, x_idx, k_idx = positive_indices[i]  # first element is sample number, which is always 0: (0, y, x, k)
    box_params = y_predicted_regression[0, y_idx, x_idx, k_idx*4+0 : k_idx*4+4]
    anchor_box = anchor_boxes[y_idx, x_idx, k_idx*4+0 : k_idx*4+4]
    proposals[i,0:4] = convert_parameterized_box_to_points(box_params = box_params, anchor_center_y = anchor_box[0], anchor_center_x = anchor_box[1], anchor_height = anchor_box[2], anchor_width = anchor_box[3])
    proposals[i,4] = y_predicted_class[0, y_idx, x_idx, k_idx]

  # Limit to max_proposals
  if max_proposals > 0:
    sorted_indices = np.argsort(proposals[:,4])                     # sorts in ascending order of score
    proposals = proposals[sorted_indices][-1:-(max_proposals+1):-1] # grab the top-N scores in descending order

  # Perform NMS to cull redundant proposals
  proposal_indices = nms(proposals = proposals, iou_threshold = 0.7)
  proposals = proposals[proposal_indices]

  # Return results clipped to image boundaries
  return _clip_box_coordinates_to_map_boundaries(boxes = proposals, map_shape = input_image_shape)

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

def _clip_box_coordinates_to_map_boundaries(boxes, map_shape):
  """
  Clips an array of image-space boxes provided as an (Nx4) tensor against the
  image boundaries.

  Each box consists of:
  0: y_min
  1: x_min
  2: y_max
  3: x_max

  Additional elements are preserved.
  """
  # First, remove boxes that are entirely out of bounds
  out_of_bounds = (boxes[:,0] >= map_shape[0]) + (boxes[:,2] < 0) + \
                  (boxes[:,1] >= map_shape[1]) + (boxes[:,3] < 0)
  invalid_indices = np.squeeze(np.argwhere(out_of_bounds))  # rows (boxes) that are entirely out of bounds and must be removed
  boxes = np.delete(boxes, invalid_indices, axis = 0)

  # Next, clip to boundaries
  y_max = map_shape[0] - 1
  x_max = map_shape[1] - 1
  boxes[:,0:4] = np.maximum(boxes[:,0:4], [ 0, 0, 0, 0 ])                 # clip to x=0 and y=0
  boxes[:,0:4] = np.minimum(boxes[:,0:4], [ y_max, x_max, y_max, x_max ]) # clip to maximum dimension

  return boxes

def label_proposals(proposals, ground_truth_object_boxes, num_classes, min_iou_threshold, max_iou_threshold):
  # One hot encoded class labels
  num_proposals = proposals.shape[0]
  y_class_labels = np.zeros((num_proposals, num_classes))

  # Regression targets and inclusion mask for each proposal and class. Shape
  # (N,2,4*(C-1)) where [n,0,:] comprises a mask for the corresponding targets
  # at [n,1,:]. Targets are ordered: ty, tx, th, tw. Background class 0 does
  # not have a box and is therefore excluded. Class index 1 targets are
  # therefore stored at [n,1,0*4:0*4+4].
  y_regression_labels = np.zeros((num_proposals, 2, 4 * (num_classes - 1)))

  # Precompute proposal center points and dimensions
  proposal_center_y = 0.5 * (proposals[:,0] + proposals[:,2])
  proposal_center_x = 0.5 * (proposals[:,1] + proposals[:,3])
  proposal_height = proposals[:,2] - proposals[:,0] + 1
  proposal_width = proposals[:,3] - proposals[:,1] + 1

  # This will determine which proposals make it through (those below the
  # minimum IoU threshold are filtered out)
  valid_indices = [] 

  # Test each proposal against each box to find the best match
  for i in range(num_proposals):
    best_iou = 0
    best_class_idx = 0                # background
    best_box = None
    passed_min_iou_threshold = False  # whether this proposal had IoU > min. threshold for any box

    for box in ground_truth_object_boxes:
      proposal_box_coords = proposals[i,0:4]
      object_box_coords = box.corners
      iou = intersection_over_union(box1 = proposal_box_coords, box2 = object_box_coords)
      if iou >= max_iou_threshold and iou > best_iou:
        best_iou = iou
        best_class_idx = box.class_index
        best_box = box
      elif iou >= min_iou_threshold:
        passed_min_iou_threshold = True      

    # Create one-hot encoded label for this proposal
    y_class_labels[i,best_class_idx] = 1.0

    # Create regression targets if we have a box. At this stage, we do not have
    # anchors and use the proposal itself as the reference. The proposals will
    # change shape during the learning process and the model will learn how to
    # transform a proposal box into an accurate bounding box.
    if best_class_idx > 0 and best_box is not None:
      box_center_y = 0.5 * (best_box.corners[0] + best_box.corners[2])  # 0.5 * (y_min + y_max)
      box_center_x = 0.5 * (best_box.corners[1] + best_box.corners[3])  # 0.5 * (x_min + x_max)
      box_height = best_box.corners[2] - best_box.corners[0] + 1        # y_max - y_min + 1
      box_width = best_box.corners[3] - best_box.corners[1] + 1         # x_max - x_min + 1
      ty = (box_center_y - proposal_center_y[i]) / proposal_height[i]
      tx = (box_center_x - proposal_center_x[i]) / proposal_width[i]
      th = log(box_height / proposal_height[i])
      tw = log(box_width / proposal_width[i])
      index = best_class_idx - 1
      y_regression_labels[i,0,4*index:4*index+4] = 1, 1, 1, 1 # mask indicating which targets are valid
      y_regression_labels[i,1,4*index:4*index+4] = ty, tx, th, tw
      valid_indices.append(i) # include this sample
    elif passed_min_iou_threshold:
      valid_indices.append(i)

  # Return only the included proposals
  return proposals[valid_indices], y_class_labels[valid_indices], y_regression_labels[valid_indices]
