#
# Faster R-CNN in PyTorch and TensorFlow 2 w/ Keras
# tf2/FasterRCNN/models/anchors.py
# Copyright 2021-2022 Bart Trzynadlowski
# 
# Anchor generation code for the TensorFlow implementation of Faster R-CNN.
# This differs from the PyTorch version only in the ordering of the dimensions
# of the input image: Keras is (height, width, channels) and PyTorch is
# (channels, height, width).
#
# Differing from other implementations of Faster R-CNN, I generate a multi-
# dimensional ground truth tensor for the RPN stage that contains a flag 
# indicating whether the anchor should be included in training, whether it is
# an object, and the box delta regression targets. It would be simpler and more
# performant to simply return 2D tensors with this information (the model ends
# up converting proposals into lists at a later stage anyway) but this is how
# I first thought to implement it and did not encounter a pressing need to
# change it.
#

import itertools
from math import sqrt
import numpy as np

from . import math_utils


def _compute_anchor_sizes():
  #
  # Anchor scales and aspect ratios.
  #
  # x * y = area          x * (x_aspect * x) = x_aspect * x^2 = area
  # x_aspect * x = y  ->  x = sqrt(area / x_aspect)
  #                       y = x_aspect * sqrt(area / x_aspect)
  #
  areas = [ 128*128, 256*256, 512*512 ]   # pixels
  x_aspects = [ 0.5, 1.0, 2.0 ]           # x:1 ratio

  # Generate all 9 combinations of area and aspect ratio
  heights = np.array([ x_aspects[j] * sqrt(areas[i] / x_aspects[j]) for (i, j) in itertools.product(range(3), range(3)) ])
  widths = np.array([ sqrt(areas[i] / x_aspects[j]) for (i, j) in itertools.product(range(3), range(3)) ])

  # Return as (9,2) matrix of sizes
  return np.vstack([ heights, widths ]).T

def generate_anchor_maps(image_shape, feature_pixels): 
  """
  Generates maps defining the anchors for a given input image size. There are 9
  different anchors at each feature map cell (3 scales, 3 ratios).

  Parameters
  ----------
  image_shape : Tuple[int, int, int]
    Shape of the input image, (height, width, channels), at the scale it will
    be passed into the Faster R-CNN model.
  feature_pixels : int
    Distance in pixels between anchors. This is the size, in input image space,
    of each cell of the feature map output by the feature extractor stage of
    the Faster R-CNN network.

  Returns
  -------
  np.ndarray, np.ndarray
    Two maps, with height and width corresponding to the feature map
    dimensions, not the input image:
      1. A map of shape (height, width, num_anchors*4) containing all anchors,
         each stored as (center_y, center_x, anchor_height, anchor_width) in
         input image pixel space.
      2. A map of shape (height, width, num_anchors) indicating which anchors
         are valid (1) or invalid (0). Invalid anchors are those that cross
         image boundaries and must not be used during training.
  """

  assert len(image_shape) == 3

  #
  # Note that precision can strongly affect anchor labeling in some images.
  # Conversion of both operands to float32 matches the implementation by Yun
  # Chen. That is, changing the final line so as to eliminate the conversion to
  # float32:
  #
  #   return anchor_map, anchor_valid_map
  #
  # Has a pronounced effect on positive anchors in image 2008_000028.jpg in
  # VOC2012.
  #
  
  # Base anchor template: (num_anchors,4), with each anchor being specified by
  # its corners (y1,x1,y2,x2)
  anchor_sizes = _compute_anchor_sizes()
  num_anchors = anchor_sizes.shape[0]
  anchor_template = np.empty((num_anchors, 4))
  anchor_template[:,0:2] = -0.5 * anchor_sizes  # y1, x1 (top-left)
  anchor_template[:,2:4] = +0.5 * anchor_sizes  # y2, x2 (bottom-right)

  # Shape of map, (H,W), determined by VGG-16 backbone
  height, width = image_shape[0] // feature_pixels, image_shape[1] // feature_pixels

  # Generate (H,W,2) map of coordinates, in feature space, each being [y,x]
  y_cell_coords = np.arange(height)
  x_cell_coords = np.arange(width)
  cell_coords = np.array(np.meshgrid(y_cell_coords, x_cell_coords)).transpose([2, 1, 0])

  # Convert all coordinates to image space (pixels) at *center* of each cell
  center_points = cell_coords * feature_pixels + 0.5 * feature_pixels

  # (H,W,2) -> (H,W,4), repeating the last dimension so it contains (y,x,y,x)
  center_points = np.tile(center_points, reps = 2)

  # (H,W,4) -> (H,W,4*num_anchors)
  center_points = np.tile(center_points, reps = num_anchors)
  
  #
  # Now we can create the anchors by adding the anchor template to each cell
  # location. Anchor template is flattened to size num_anchors * 4 to make 
  # the addition possible (along the last dimension). 
  #
  anchors = center_points.astype(np.float32) + anchor_template.flatten()

  # (H,W,4*num_anchors) -> (H*W*num_anchors,4)
  anchors = anchors.reshape((height*width*num_anchors, 4))

  # Valid anchors are those that do not cross image boundaries
  image_height, image_width = image_shape[0:2]
  valid = np.all((anchors[:,0:2] >= [0,0]) & (anchors[:,2:4] <= [image_height,image_width]), axis = 1)

  # Convert anchors to anchor format: (center_y, center_x, height, width)
  anchor_map = np.empty((anchors.shape[0], 4))
  anchor_map[:,0:2] = 0.5 * (anchors[:,0:2] + anchors[:,2:4])
  anchor_map[:,2:4] = anchors[:,2:4] - anchors[:,0:2]

  # Reshape maps and return
  anchor_map = anchor_map.reshape((height, width, num_anchors * 4))
  anchor_valid_map = valid.reshape((height, width, num_anchors))
  return anchor_map.astype(np.float32), anchor_valid_map.astype(np.float32)

def generate_rpn_map(anchor_map, anchor_valid_map, gt_boxes, object_iou_threshold = 0.7, background_iou_threshold = 0.3):
  """
  Generates a map containing ground truth data for training the region proposal
  network.

  Parameters
  ----------
  anchor_map : np.ndarray
    Map of shape (height, width, num_anchors*4) defining the anchors as
    (center_y, center_x, anchor_height, anchor_width) in input image space.
  anchor_valid_map : np.ndarray
    Map of shape (height, width, num_anchors) defining anchors that are valid
    and may be included in training.
  gt_boxes : List[training_sample.Box]
    List of ground truth boxes.
  object_iou_threshold : float
    IoU threshold between an anchor and a ground truth box above which an
    anchor is labeled as an object (positive) anchor.
  background_iou_threshold : float
    IoU threshold below which an anchor is labeled as background (negative).

  Returns
  -------
  np.ndarray, np.ndarray, np.ndarray
    RPN ground truth map, object (positive) anchor indices, and background
    (negative) anchor indices. Map height and width dimensions are in feature
    space.
    1. RPN ground truth map of shape (height, width, num_anchors, 6) where the
       last dimension is:
       - 0: Trainable anchor (1) or not (0). Only valid and non-neutral (that
            is, definitely positive or negative) anchors are trainable. This is
            the same as anchor_valid_map with additional invalid anchors caused
            by neutral samples
       - 1: For trainable anchors, whether the anchor is an object anchor (1)
            or background anchor (0). For non-trainable anchors, will be 0.
       - 2: Regression target for box center, ty.
       - 3: Regression target for box center, tx.
       - 4: Regression target for box size, th.
       - 5: Regression target for box size, tw.
    2. Map of shape (N, 3) of indices (y, x, k) of all N object anchors in the
       RPN ground truth map.
    3. Map of shape (M, 3) of indices of all M background anchors in the RPN
       ground truth map.
  """
  height, width, num_anchors = anchor_valid_map.shape

  # Convert ground truth box corners to (M,4) tensor and class indices to (M,)
  gt_box_corners = np.array([ box.corners for box in gt_boxes ])
  num_gt_boxes = len(gt_boxes)

  # Compute ground truth box center points and side lengths
  gt_box_centers = 0.5 * (gt_box_corners[:,0:2] + gt_box_corners[:,2:4])
  gt_box_sides = gt_box_corners[:,2:4] - gt_box_corners[:,0:2]

  # Flatten anchor boxes to (N,4) and convert to corners
  anchor_map = anchor_map.reshape((-1,4))
  anchors = np.empty(anchor_map.shape)
  anchors[:,0:2] = anchor_map[:,0:2] - 0.5 * anchor_map[:,2:4]  # y1, x1
  anchors[:,2:4] = anchor_map[:,0:2] + 0.5 * anchor_map[:,2:4]  # y2, x2
  n = anchors.shape[0]

  # Initialize all anchors initially as negative (background). We will also
  # track which ground truth box was assigned to each anchor.
  objectness_score = np.full(n, -1)   # RPN class: 0 = background, 1 = foreground, -1 = ignore (these will be marked as invalid in the truth map)
  gt_box_assignments = np.full(n, -1) # -1 means no box
  
  # Compute IoU between each anchor and each ground truth box, (N,M).
  ious = math_utils.intersection_over_union(boxes1 = anchors, boxes2 = gt_box_corners)

  # Need to remove anchors that are invalid (straddle image boundaries) from
  # consideration entirely and the easiest way to do this is to wipe out their
  # IoU scores
  ious[anchor_valid_map.flatten() == 0, :] = -1.0

  # Find the best IoU ground truth box for each anchor and the best IoU anchor
  # for each ground truth box.
  #
  # Note that ious == max_iou_per_gt_box tests each of the N rows of ious
  # against the M elements of max_iou_per_gt_box, column-wise. np.where() then
  # returns all (y,x) indices of matches as a tuple: (y_indices, x_indices).
  # The y indices correspond to the N dimension and therefore indicate anchors
  # and the x indices correspond to the M dimension (ground truth boxes).
  max_iou_per_anchor = np.max(ious, axis = 1)           # (N,)
  best_box_idx_per_anchor = np.argmax(ious, axis = 1)   # (N,)
  max_iou_per_gt_box = np.max(ious, axis = 0)           # (M,)
  highest_iou_anchor_idxs = np.where(ious == max_iou_per_gt_box)[0] # get (L,) indices of anchors that are the highest-overlapping anchors for at least one of the M boxes

  # Anchors below the minimum threshold are negative
  objectness_score[max_iou_per_anchor < background_iou_threshold] = 0

  # Anchors that meet the threshold IoU are positive
  objectness_score[max_iou_per_anchor >= object_iou_threshold] = 1

  # Anchors that overlap the most with ground truth boxes are positive
  objectness_score[highest_iou_anchor_idxs] = 1

  # We assign the highest IoU ground truth box to each anchor. If no box met
  # the IoU threshold, the highest IoU box may happen to be a box for which
  # the anchor had the highest IoU. If not, then the objectness score will be
  # negative and the box regression won't ever be used.
  gt_box_assignments[:] = best_box_idx_per_anchor

  # Anchors that are to be ignored will be marked invalid. Generate a mask to
  # multiply anchor_valid_map by (-1 -> 0, 0 or 1 -> 1). Then mark ignored
  # anchors as 0 in objectness score because the score can only really be 0 or
  # 1.
  enable_mask = (objectness_score >= 0).astype(np.float32)
  objectness_score[objectness_score < 0] = 0
  
  # Compute box delta regression targets for each anchor
  box_delta_targets = np.empty((n, 4))
  box_delta_targets[:,0:2] = (gt_box_centers[gt_box_assignments] - anchor_map[:,0:2]) / anchor_map[:,2:4] # ty = (box_center_y - anchor_center_y) / anchor_height, tx = (box_center_x - anchor_center_x) / anchor_width
  box_delta_targets[:,2:4] = np.log(gt_box_sides[gt_box_assignments] / anchor_map[:,2:4])                 # th = log(box_height / anchor_height), tw = log(box_width / anchor_width)

  # Assemble RPN ground truth map
  rpn_map = np.zeros((height, width, num_anchors, 6))
  rpn_map[:,:,:,0] = anchor_valid_map * enable_mask.reshape((height,width,num_anchors))  # trainable anchors (object or background; excludes boundary-crossing invalid and neutral anchors)
  rpn_map[:,:,:,1] = objectness_score.reshape((height,width,num_anchors))
  rpn_map[:,:,:,2:6] = box_delta_targets.reshape((height,width,num_anchors,4))
  
  # Return map along with positive and negative anchors
  rpn_map_coords = np.transpose(np.mgrid[0:height,0:width,0:num_anchors], (1,2,3,0))                  # shape (height,width,k,3): every index (y,x,k,:) returns its own coordinate (y,x,k)
  object_anchor_idxs = rpn_map_coords[np.where((rpn_map[:,:,:,1] > 0) & (rpn_map[:,:,:,0] > 0))]      # shape (N,3), where each row is the coordinate (y,x,k) of a positive sample
  background_anchor_idxs = rpn_map_coords[np.where((rpn_map[:,:,:,1] == 0) & (rpn_map[:,:,:,0] > 0))] # shape (N,3), where each row is the coordinate (y,x,k) of a negative sample

  return rpn_map.astype(np.float32), object_anchor_idxs, background_anchor_idxs
