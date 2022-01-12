#
# FasterRCNN in PyTorch and TensorFlow 2 w/ Keras
# python/tf2/FasterRCNN/models/math_utils.py
# Copyright 2021-2022 Bart Trzynadlowski
#
# Math helper functions.
#

import numpy as np


def intersection_over_union(boxes1, boxes2):
  """
  Computes intersection-over-union (IoU) for multiple boxes in parallel.

  Parameters
  ----------
  boxes1 : np.ndarray
    Box corners, shaped (N, 4), with each box as (y1, x1, y2, x2).
  boxes2 : np.ndarray
    Box corners, shaped (M, 4).

  Returns
  -------
  np.ndarray
    IoUs for each pair of boxes in boxes1 and boxes2, shaped (N, M).
  """
  top_left_point = np.maximum(boxes1[:,None,0:2], boxes2[:,0:2])                                  # (N,1,2) and (M,2) -> (N,M,2) indicating top-left corners of box pairs
  bottom_right_point = np.minimum(boxes1[:,None,2:4], boxes2[:,2:4])                              # "" bottom-right corners ""
  well_ordered_mask = np.all(top_left_point < bottom_right_point, axis = 2)                       # (N,M) indicating whether top_left_x < bottom_right_x and top_left_y < bottom_right_y (meaning boxes may intersect)
  intersection_areas = well_ordered_mask * np.prod(bottom_right_point - top_left_point, axis = 2) # (N,M) indicating intersection area (bottom_right_x - top_left_x) * (bottom_right_y - top_left_y)
  areas1 = np.prod(boxes1[:,2:4] - boxes1[:,0:2], axis = 1)                                       # (N,) indicating areas of boxes1
  areas2 = np.prod(boxes2[:,2:4] - boxes2[:,0:2], axis = 1)                                       # (M,) indicating areas of boxes2
  union_areas = areas1[:,None] + areas2 - intersection_areas                                      # (N,1) + (M,) - (N,M) = (N,M), union areas of both boxes
  epsilon = 1e-7
  return intersection_areas / (union_areas + epsilon)

def convert_regressions_to_boxes(regressions, anchors, regression_means, regression_stds):
  """
  Converts regressions, which are in parameterized form (ty, tx, th, tw) as
  described by the FastRCNN and FasterRCNN papers, to boxes (y1, x1, y2, x2).
  The anchors are the base boxes (e.g., RPN anchors or proposals) that the
  regressions describe a modification to.

  Parameters
  ----------
  regressions : np.ndarray
    Regression parameters with shape (N, 4). Each row is (ty, tx, th, tw).
  anchors : np.ndarray
    Corresponding anchors that the regressed parameters are based upon,
    shaped (N, 4) with each row being (center_y, center_x, height, width).
  regression_means : np.ndarray
    Mean ajustment to regressions, (4,), to be added after standard deviation
    scaling and before conversion to actual box coordinates.
  regression_stds : np.ndarray
    Standard deviation adjustment to regressions, (4,). Regression parameters
    are first multiplied by these values.

  Returns
  -------
  np.ndarray
    Box coordinates, (N, 4), with each row being (y1, x1, y2, x2).
  """
  regressions = regressions * regression_stds + regression_means
  center = anchors[:,2:4] * regressions[:,0:2] + anchors[:,0:2] # center_x = anchor_width * tx + anchor_center_x, center_y = anchor_height * ty + anchor_center_y
  size = anchors[:,2:4] * np.exp(regressions[:,2:4])            # width = anchor_width * exp(tw), height = anchor_height * exp(th)
  boxes = np.empty(regressions.shape)
  boxes[:,0:2] = center - 0.5 * size                            # y1, x1
  boxes[:,2:4] = center + 0.5 * size                            # y2, x2
  return boxes
