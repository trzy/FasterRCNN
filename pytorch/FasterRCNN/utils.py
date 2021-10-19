import numpy as np


def parallel_intersection_over_union(boxes1, boxes2):
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
