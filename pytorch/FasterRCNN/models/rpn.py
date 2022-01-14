#
# FasterRCNN in PyTorch and TensorFlow 2 w/ Keras
# pytorch/FasterRCNN/models/rpn.py
# Copyright 2021-2022 Bart Trzynadlowski
# 
# PyTorch implementation of the RPN (region proposal network) stage of
# FasterRCNN. Given a feature map (i.e., the output of the VGG-16 convolutional
# layers), generates objectness scores for each anchor box, and boxes in the
# form of modifications to anchor center points and dimensions.
#
# The RPN class and box regression losses are defined here.
#

import numpy as np
import torch as t
from torch import nn
from torch.nn import functional as F
from torchvision.ops import nms

from . import math_utils


class RegionProposalNetwork(nn.Module):
  def __init__(self, allow_edge_proposals = False):
    super().__init__()

    # Constants
    self._allow_edge_proposals = allow_edge_proposals

    # Layers
    num_anchors = 9
    self._rpn_conv1 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = (3, 3), stride = 1, padding = "same")
    self._rpn_class = nn.Conv2d(in_channels = 512, out_channels = num_anchors, kernel_size = (1, 1), stride = 1, padding = "same")
    self._rpn_boxes = nn.Conv2d(in_channels = 512, out_channels = num_anchors * 4, kernel_size = (1, 1), stride = 1, padding = "same")
    
    # Initialize weights
    self._rpn_conv1.weight.data.normal_(mean = 0.0, std = 0.01)
    self._rpn_conv1.bias.data.zero_()
    self._rpn_class.weight.data.normal_(mean = 0.0, std = 0.01)
    self._rpn_class.bias.data.zero_()
    self._rpn_boxes.weight.data.normal_(mean = 0.0, std = 0.01)
    self._rpn_boxes.bias.data.zero_()

  def forward(self, feature_map, image_shape, anchor_map, anchor_valid_map, max_proposals_pre_nms, max_proposals_post_nms):
    """
    Predict objectness scores and regress region-of-interest box proposals on
    an input feature map.

    Parameters
    ----------
    feature_map : torch.Tensor
      Feature map of shape (batch_size, 512, height, width).
    image_shape : Tuple[int, int, int]
      Shapes of each image in pixels: (num_channels, height, width).
    anchor_map : np.ndarray
      Map of anchors, shaped (height, width, num_anchors * 4). The last
      dimension contains the anchor boxes specified as a 4-tuple of
      (center_y, center_x, height, width), repeated for all anchors at that
      coordinate of the feature map.
    anchor_valid_map : np.ndarray
      Map indicating which anchors are valid (do not intersect image bounds),
      shaped (height, width, num_anchors).
    max_proposals_pre_nms : int
      How many of the best proposals (sorted by objectness score) to extract
      before applying non-maximum suppression.
    max_proposals_post_nms : int
      How many of the best proposals (sorted by objectness score) to keep after
      non-maximum suppression.

    Returns
    -------
    torch.Tensor, torch.Tensor, np.ndarray
      - Objectness scores (batch_size, height, width, num_anchors)
      - Box regressions (batch_size, height, width, num_anchors * 4), in
        parameterized form (that is, (ty, tx, th, tw) for each anchor)
      - Proposals (N, 4) -- all corresponding proposal box corners stored as
        (y1, x1, y2, x2).
    """

    # Pass through the network
    y = F.relu(self._rpn_conv1(feature_map))
    objectness_score_map = t.sigmoid(self._rpn_class(y))
    box_regression_map = self._rpn_boxes(y)

    # Transpose shapes to be more convenient:
    #   objectness_score_map -> (batch_size, height, width, num_anchors)
    #   box_regression_map   -> (batch_size, height, width, num_anchors * 4)
    objectness_score_map = objectness_score_map.permute(0, 2, 3, 1).contiguous()
    box_regression_map = box_regression_map.permute(0, 2, 3, 1).contiguous()

    # Returning to CPU land by extracting proposals as lists (NumPy arrays)
    anchors, objectness_scores, box_regressions = self._extract_valid(
      anchor_map = anchor_map,
      anchor_valid_map = anchor_valid_map,
      objectness_score_map = objectness_score_map,
      box_regression_map = box_regression_map
    )

    # Convert regressions to box corners
    proposals = math_utils.convert_regressions_to_boxes(regressions = box_regressions, anchors = anchors, regression_means = [ 0, 0, 0, 0 ], regression_stds = [ 1, 1, 1, 1 ]).astype(np.float32)

    # Keep only the top-N scores. Note that we do not care whether the
    # proposals were labeled as objects (score > 0.5) and peform a simple
    # ranking among all of them. Restricting them has a strong adverse impact
    # on training performance.
    sorted_indices = np.argsort(objectness_scores)                  # sort in ascending order of objectness score
    sorted_indices = sorted_indices[::-1]                           # descending order of score
    proposals = proposals[sorted_indices][0:max_proposals_pre_nms]  # grab the top-N best proposals
    objectness_scores = objectness_scores[sorted_indices][0:max_proposals_pre_nms]  # corresponding scores

    # Clip to image boundaries
    proposals[:,0:2] = np.maximum(proposals[:,0:2], 0)
    proposals[:,2] = np.minimum(proposals[:,2], image_shape[1])
    proposals[:,3] = np.minimum(proposals[:,3], image_shape[2])

    # Remove anything less than 16 pixels on a side
    height = proposals[:,2] - proposals[:,0]
    width = proposals[:,3] - proposals[:,1]
    idxs = np.where((height >= 16) & (width >= 16))[0]
    proposals = proposals[idxs]
    objectness_scores = objectness_scores[idxs]

    # Perform NMS
    idxs = nms(
      boxes = t.from_numpy(proposals).cuda(),
      scores = t.from_numpy(objectness_scores).cuda(),
      iou_threshold = 0.7
    )
    idxs = idxs[0:max_proposals_post_nms].cpu().numpy()
    proposals = proposals[idxs]

    # Return network outputs as PyTorch tensors and extracted object proposals
    # as NumPy arrays
    return objectness_score_map, box_regression_map, proposals

  def _extract_valid(self, anchor_map, anchor_valid_map, objectness_score_map, box_regression_map):
    assert objectness_score_map.shape[0] == 1 # only batch size of 1 supported for now

    height, width, num_anchors = anchor_valid_map.shape
    anchors = anchor_map.reshape((height * width * num_anchors, 4))             # [N,4] all anchors
    anchors_valid = anchor_valid_map.reshape((height * width * num_anchors))    # [N,] whether anchors are valid (i.e., do not cross image boundaries)
    scores = objectness_score_map.reshape((height * width * num_anchors))       # [N,] prediced objectness scores
    regressions = box_regression_map.reshape((height * width * num_anchors, 4)) # [N,4] predicted regression targets

    if self._allow_edge_proposals:
      # Use all proposals
      return anchors, scores.cpu().detach().numpy(), regressions.cpu().detach().numpy()
    else:
      # Filter out those proposals generated at invalid anchors
      idxs = anchors_valid > 0
      return anchors[idxs], scores[idxs].cpu().numpy(), regressions[idxs].cpu().numpy()


def class_loss(predicted_scores, y_true):
  """
  Computes RPN class loss.

  Parameters
  ----------
  predicted_scores : torch.Tensor
    A tensor of shape (batch_size, height, width, num_anchors) containing
    objectness scores (0 = background, 1 = object).
  y_true : torch.Tensor
    Ground truth tensor of shape (batch_size, height, width, num_anchors, 6).

  Returns
  -------
  torch.Tensor
    Scalar loss.
  """

  epsilon = 1e-7

  # y_true_class: (batch_size, height, width, num_anchors), same as predicted_scores
  y_true_class = y_true[:,:,:,:,1].reshape(predicted_scores.shape)
  y_predicted_class = predicted_scores
  
  # y_mask: y_true[:,:,:,0] is 1.0 for anchors included in the mini-batch
  y_mask = y_true[:,:,:,:,0].reshape(predicted_scores.shape)

  # Compute how many anchors are actually used in the mini-batch (e.g.,
  # typically 256)
  N_cls = t.count_nonzero(y_mask) + epsilon

  # Compute element-wise loss for all anchors
  loss_all_anchors = F.binary_cross_entropy(input = y_predicted_class, target = y_true_class, reduction = "none")
  
  # Zero out the ones which should not have been included
  relevant_loss_terms = y_mask * loss_all_anchors

  # Sum the total loss and normalize by the number of anchors used
  return t.sum(relevant_loss_terms) / N_cls

def regression_loss(predicted_regressions, y_true):
  """
  Computes RPN regression loss.

  Parameters
  ----------
  predicted_regressions : torch.Tensor
    A tensor of shape (batch_size, height, width, num_anchors * 4) containing
    RoI box regressions for each anchor, stored as: ty, tx, th, tw.
  y_true : torch.Tensor
    Ground truth tensor of shape (batch_size, height, width, num_anchors, 6).

  Returns
  -------
  torch.Tensor
    Scalar loss.
  """

  epsilon = 1e-7
  scale_factor = 1.0  # hyper-parameter that controls magnitude of regression loss and is chosen to make regression term comparable to class term
  sigma = 3.0         # see: https://github.com/rbgirshick/py-faster-rcnn/issues/89
  sigma_squared = sigma * sigma

  y_predicted_regression = predicted_regressions
  y_true_regression = y_true[:,:,:,:,2:6].reshape(y_predicted_regression.shape)

  # Include only anchors that are used in the mini-batch and which correspond
  # to objects (positive samples)
  y_included = y_true[:,:,:,:,0].reshape(y_true.shape[0:4]) # trainable anchors map: (batch_size, height, width, num_anchors)
  y_positive = y_true[:,:,:,:,1].reshape(y_true.shape[0:4]) # positive anchors
  y_mask = y_included * y_positive

  # y_mask is of the wrong shape. We have one value per (y,x,k) position but in
  # fact need to have 4 values (one for each of the regression variables). For
  # example, y_predicted might be (1,37,50,36) and y_mask will be (1,37,50,9).
  # We need to repeat the last dimension 4 times.
  y_mask = y_mask.repeat_interleave(repeats = 4, dim = 3)

  # The paper normalizes by dividing by a quantity called N_reg, which is equal
  # to the total number of anchors (~2400) and then multiplying by lambda=10.
  # This does not make sense to me because we are summing over a mini-batch at
  # most, so we use N_cls here. I might be misunderstanding what is going on
  # but 10/2400 = 1/240 which is pretty close to 1/256 and the paper mentions
  # that training is relatively insensitve to choice of normalization.
  N_cls = t.count_nonzero(y_included) + epsilon

  # Compute element-wise loss using robust L1 function for all 4 regression
  # components
  x = y_true_regression - y_predicted_regression
  x_abs = t.abs(x)
  is_negative_branch = (x_abs < (1.0 / sigma_squared)).float()
  R_negative_branch = 0.5 * x * x * sigma_squared
  R_positive_branch = x_abs - 0.5 / sigma_squared
  loss_all_anchors = is_negative_branch * R_negative_branch + (1.0 - is_negative_branch) * R_positive_branch

  # Zero out the ones which should not have been included
  relevant_loss_terms = y_mask * loss_all_anchors
  return scale_factor * t.sum(relevant_loss_terms) / N_cls
