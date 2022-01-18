#
# Faster R-CNN in PyTorch and TensorFlow 2 w/ Keras
# tf2/FasterRCNN/models/math_utils.py
# Copyright 2021-2022 Bart Trzynadlowski
#
# Math helper functions.
#

import numpy as np
import tensorflow as tf


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

def tf_intersection_over_union(boxes1, boxes2):
  """
  Equivalent of intersection_over_union() but operates on tf.Tensors and
  produces a TensorFlow graph suitable for use in a model. This code borrowed
  from Matterport's MaskRCNN implementation:
  https://github.com/matterport/Mask_RCNN

  Parameters
  ----------
  boxes1: tf.Tensor
    Box corners, shaped (N,4), with each box as (y1, x1, y2, x2).
  boxes2: tf.Tensor
    Box corners, shaped (M,4).

  Returns
  -------
  tf.Tensor
    Tensor of shape (N, M) containing IoU score between each pair of boxes.
  """
  # 1. Tile boxes2 and repeat boxes1. This allows us to compare
  # every boxes1 against every boxes2 without loops.
  # TF doesn't have an equivalent to np.repeat() so simulate it
  # using tf.tile() and tf.reshape.
  b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1),
                          [1, 1, tf.shape(boxes2)[0]]), [-1, 4])
  b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])
  # 2. Compute intersections
  b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
  b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
  y1 = tf.maximum(b1_y1, b2_y1)
  x1 = tf.maximum(b1_x1, b2_x1)
  y2 = tf.minimum(b1_y2, b2_y2)
  x2 = tf.minimum(b1_x2, b2_x2)
  intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
  # 3. Compute unions
  b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
  b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
  union = b1_area + b2_area - intersection
  # 4. Compute IoU and reshape to [boxes1, boxes2]
  iou = intersection / union
  overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
  return overlaps

def convert_deltas_to_boxes(box_deltas, anchors, box_delta_means, box_delta_stds):
  """
  Converts box deltas, which are in parameterized form (ty, tx, th, tw) as
  described by the Fast R-CNN and Faster R-CNN papers, to boxes
  (y1, x1, y2, x2). The anchors are the base boxes (e.g., RPN anchors or
  proposals) that the deltas describe a modification to.

  Parameters
  ----------
  box_deltas : np.ndarray
    Box deltas with shape (N, 4). Each row is (ty, tx, th, tw).
  anchors : np.ndarray
    Corresponding anchors that the deltas are based upon, shaped (N, 4) with
    each row being (center_y, center_x, height, width).
  box_delta_means : np.ndarray
    Mean ajustment to deltas, (4,), to be added after standard deviation
    scaling and before conversion to actual box coordinates.
  box_delta_stds : np.ndarray
    Standard deviation adjustment to deltas, (4,). Box deltas are first
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

def tf_convert_deltas_to_boxes(box_deltas, anchors, box_delta_means, box_delta_stds):
  """
  Equivalent of convert_deltas_to_boxes() but operates on tf.Tensors and
  produces a TensorFlow graph suitable for use in a model.

  Parameters
  ----------
  box_deltas : np.ndarray
    Box deltas with shape (N, 4). Each row is (ty, tx, th, tw).
  anchors : np.ndarray
    Corresponding anchors that the deltas are based upon, shaped (N, 4) with
    each row being (center_y, center_x, height, width).
  box_delta_means : np.ndarray
    Mean ajustment to deltas, (4,), to be added after standard deviation
    scaling and before conversion to actual box coordinates.
  box_delta_stds : np.ndarray
    Standard deviation adjustment to deltas, (4,). Box deltas are first
    multiplied by these values.

  Returns
  -------
  tf.Tensor
    Box coordinates, (N, 4), with each row being (y1, x1, y2, x2).
  """
  box_deltas = box_deltas * box_delta_stds + box_delta_means
  center = anchors[:,2:4] * box_deltas[:,0:2] + anchors[:,0:2]  # center_x = anchor_width * tx + anchor_center_x, center_y = anchor_height * ty + anchor_center_y
  size = anchors[:,2:4] * tf.math.exp(box_deltas[:,2:4])        # width = anchor_width * exp(tw), height = anchor_height * exp(th)
  boxes_top_left = center - 0.5 * size                          # y1, x1
  boxes_bottom_right = center + 0.5 * size                      # y2, x2
  boxes = tf.concat([ boxes_top_left, boxes_bottom_right ], axis = 1) # [ (N,2), (N,2) ] -> (N,4)
  return boxes
