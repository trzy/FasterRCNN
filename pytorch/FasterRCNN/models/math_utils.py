#
# Faster R-CNN in PyTorch and TensorFlow 2 w/ Keras
# pytorch/FasterRCNN/models/math_utils.py
# Copyright 2021-2022 Bart Trzynadlowski
#
# Math helper functions.
#

import numpy as np
import torch as t


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

def t_intersection_over_union(boxes1, boxes2):
  """
  Equivalent to intersection_over_union(), operating on PyTorch tensors.

  Parameters
  ----------
  boxes1 : torch.Tensor
    Box corners, shaped (N, 4), with each box as (y1, x1, y2, x2).
  boxes2 : torch.Tensor
    Box corners, shaped (M, 4).

  Returns
  -------
  torch.Tensor
    IoUs for each pair of boxes in boxes1 and boxes2, shaped (N, M).
  """
  top_left_point = t.maximum(boxes1[:,None,0:2], boxes2[:,0:2])                                 # (N,1,2) and (M,2) -> (N,M,2) indicating top-left corners of box pairs
  bottom_right_point = t.minimum(boxes1[:,None,2:4], boxes2[:,2:4])                             # "" bottom-right corners ""
  well_ordered_mask = t.all(top_left_point < bottom_right_point, axis = 2)                      # (N,M) indicating whether top_left_x < bottom_right_x and top_left_y < bottom_right_y (meaning boxes may intersect)
  intersection_areas = well_ordered_mask * t.prod(bottom_right_point - top_left_point, dim = 2) # (N,M) indicating intersection area (bottom_right_x - top_left_x) * (bottom_right_y - top_left_y)
  areas1 = t.prod(boxes1[:,2:4] - boxes1[:,0:2], dim = 1)                                       # (N,) indicating areas of boxes1
  areas2 = t.prod(boxes2[:,2:4] - boxes2[:,0:2], dim = 1)                                       # (M,) indicating areas of boxes2
  union_areas = areas1[:,None] + areas2 - intersection_areas                                    # (N,1) + (M,) - (N,M) = (N,M), union areas of both boxes
  epsilon = 1e-7
  return intersection_areas / (union_areas + epsilon)

def convert_deltas_to_boxes(box_deltas, anchors, box_delta_means, box_delta_stds):
  """
  Converts box deltas, which are in parameterized form (ty, tx, th, tw) as
  described by the Fast R-CNN and Faster R-CNN papers, to boxes
  (y1, x1, y2, x2). The anchors are the base boxes (e.g., RPN anchors or
  proposals) that the box deltas describe a modification to.

  Parameters
  ----------
  box_deltas : np.ndarray
    Box deltas with shape (N, 4). Each row is (ty, tx, th, tw).
  anchors : np.ndarray
    Corresponding anchors that the box deltas are based upon, shaped (N, 4)
    with each row being (center_y, center_x, height, width).
  box_delta_means : np.ndarray
    Mean ajustment to box deltas, (4,), to be added after standard deviation
    scaling and before conversion to actual box coordinates.
  box_delta_stds : np.ndarray
    Standard deviation adjustment to box deltas, (4,). Box deltas are first
    multiplied by these values.

  Returns
  -------
  np.ndarray
    Box coordinates, (N, 4), with each row being (y1, x1, y2, x2).
  """
  box_deltas = box_deltas * box_delta_stds + box_delta_means
  center = anchors[:,2:4] * box_deltas[:,0:2] + anchors[:,0:2]  # center_x = anchor_width * tx + anchor_center_x, center_y = anchor_height * ty + anchor_center_y
  size = anchors[:,2:4] * np.exp(box_deltas[:,2:4])             # width = anchor_width * exp(tw), height = anchor_height * exp(th)
  boxes = np.empty(box_deltas.shape)
  boxes[:,0:2] = center - 0.5 * size                            # y1, x1
  boxes[:,2:4] = center + 0.5 * size                            # y2, x2
  return boxes

def t_convert_deltas_to_boxes(box_deltas, anchors, box_delta_means, box_delta_stds):
  """
  Equivalent to convert_deltas_to_boxes(), operating on PyTorch tensors.

  Parameters
  ----------
  box_deltas : torch.Tensor
    Box deltas with shape (N, 4). Each row is (ty, tx, th, tw).
  anchors : torch.Tensor
    Corresponding anchors that the box deltas are based upon, shaped (N, 4)
    with each row being (center_y, center_x, height, width).
  box_delta_means : torch.Tensor
    Mean ajustment to box deltas, (4,), to be added after standard deviation
    scaling and before conversion to actual box coordinates.
  box_delta_stds : torch.Tensor
    Standard deviation adjustment to box deltas, (4,). Box deltas are first
    multiplied by these values.

  Returns
  -------
  torch.Tensor
    Box coordinates, (N, 4), with each row being (y1, x1, y2, x2).
  """
  box_deltas = box_deltas * box_delta_stds + box_delta_means
  center = anchors[:,2:4] * box_deltas[:,0:2] + anchors[:,0:2]  # center_x = anchor_width * tx + anchor_center_x, center_y = anchor_height * ty + anchor_center_y
  size = anchors[:,2:4] * t.exp(box_deltas[:,2:4])              # width = anchor_width * exp(tw), height = anchor_height * exp(th)
  boxes = t.empty(box_deltas.shape, dtype = t.float32, device = "cuda")
  boxes[:,0:2] = center - 0.5 * size                            # y1, x1
  boxes[:,2:4] = center + 0.5 * size                            # y2, x2
  return boxes
