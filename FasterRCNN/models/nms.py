import numpy as np

def nms(proposals, iou_threshold):
  """
  Takes proposals, of size (N, 5), where N is the number of proposals and the
  other dimension consists of:

    0: y_min
    1: x_min
    2: y_max
    3: x_max
    4: class score (0 if not object, 1 if object)

  Returns an array of indices between 0 and N-1 to keep.
  """
  # Precompute some important arrays for each proposals
  y_min = proposals[:,0]
  x_min = proposals[:,1]
  y_max = proposals[:,2]
  x_max = proposals[:,3]
  score = proposals[:,4]
  area = (x_max - x_min + 1) * (y_max - y_min + 1)

  ordered = score.argsort()[::-1] # indices of highest to lowest scores
  
  proposals = []
  while ordered.size > 0:
    # Keep the best-scoring proposal
    best_idx = ordered[0]
    proposals.append(best_idx)
    
    # Compute IoU against every other proposal box
    y1 = np.maximum(y_min[best_idx], y_min[ordered[1:]])
    x1 = np.maximum(x_min[best_idx], x_min[ordered[1:]])
    y2 = np.minimum(y_max[best_idx], y_max[ordered[1:]])
    x2 = np.minimum(x_max[best_idx], x_max[ordered[1:]])
    height = np.maximum(y2 - y1 + 1, 0.0)
    width = np.maximum(x2 - x1 + 1, 0.0)
    intersection = height * width
    union = area[best_idx] + area[ordered[1:]] - intersection
    iou = intersection / union

    # Throw away everything with IoU > threshold -- that is, keep only non-
    # redundant boxes
    indices = np.where(iou <= iou_threshold)[0]
    ordered = ordered[indices + 1]  # indices did not include first element, must map it back to full array

  return proposals
