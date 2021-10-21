#
# TODO:
# -----
# - Detector network regressions should be of shape N,(num_classes-1)*4 (eliminate the unneeded outputs for background)
# - IoU threshold for prediction. Is it 0.3 as here or 0.5? Check the paper.
#

from dataclasses import dataclass
import numpy as np
import torch as t
from torch import nn
from torchvision.ops import nms

from FasterRCNN.datasets import anchors
from FasterRCNN import utils
from . import math_utils
from . import vgg16
from . import rpn
from . import detector


class FasterRCNNModel(nn.Module):
  @dataclass
  class Loss:
    rpn_class:            t.Tensor
    rpn_regression:       t.Tensor
    detector_class:       t.Tensor
    detector_regression:  t.Tensor
    total:                t.Tensor

  def __init__(self, num_classes, proposal_batch = 128, allow_edge_proposals = True):
    super().__init__()

    # Constants
    self._num_classes = num_classes
    self._proposal_batch = proposal_batch
    self._detector_regression_means = (0, 0, 0, 0)
    self._detector_regression_stds = (0.1, 0.1, 0.2, 0.2)

    # Network stages
    self._stage1_feature_extractor = vgg16.FeatureExtractor()
    self._stage2_region_proposal_network = rpn.RegionProposalNetwork(allow_edge_proposals = allow_edge_proposals)
    self._stage3_detector_network = detector.DetectorNetwork(num_classes = num_classes)

  def forward(self, image_data, anchor_map = None, anchor_valid_map = None):
    """
    Forward inference. Use for test and evaluation only.

    Parameters
    ----------
    image_data : torch.Tensor
      A tensor of shape (batch_size, channels, height, width) representing
      images normalized using the VGG-16 convention (BGR, ImageNet channel-wise
      mean-centered).
    anchor_map : torch.Tensor
      Map of anchors, shaped (height, width, num_anchors * 4). The last
      dimension contains the anchor boxes specified as a 4-tuple of
      (center_y, center_x, height, width), repeated for all anchors at that
      coordinate of the feature map. If this or anchor_valid_map is not
      provided, both will be computed here.
    anchor_valid_map : torch.Tensor
      Map indicating which anchors are valid (do not intersect image bounds),
      shaped (height, width). If this or anchor_map is not provided, both will
      be computed here.

    Returns
    -------
    np.ndarray, torch.Tensor, torch.Tensor
      - Proposals (N, 4) from region proposal network
      - Classes (M, num_classes) from detector network
      - Box regression parameters (M, (num_classes - 0) * 4) from detector
        network
    """
    assert image_data.shape[0] == 1, "Batch size must be 1"
    image_shape = image_data.shape[1:]  # (batch_index, channels, height, width) -> (channels, height, width)

    # Anchor maps can be pre-computed and passed in explicitly (for performance
    # reasons) but if they are missing, we compute them on-the-fly here
    if anchor_map is None or anchor_valid_map is None:
      anchor_map, anchor_valid_map = anchors.generate_anchor_maps(image_shape = image_shape, feature_pixels = 16) 

    # Run each stage
    feature_map = self._stage1_feature_extractor(image_data = image_data)
    objectness_score_map, box_regression_map, proposals = self._stage2_region_proposal_network(
      feature_map = feature_map,
      image_shape = image_shape,
      anchor_map = anchor_map,
      anchor_valid_map = anchor_valid_map,
      max_proposals_pre_nms = 6000, # test time values
      max_proposals_post_nms = 300
    )
    classes, regressions = self._stage3_detector_network(
      feature_map = feature_map,
      proposals = proposals
    )

    return proposals, classes, regressions

  @utils.no_grad
  def predict(self, image_data, score_threshold, anchor_map = None, anchor_valid_map = None):
    """
    Performs inference on an image and obtains the final detected boxes.

    Parameters
    ---------- 
    image_data : torch.Tensor
      A tensor of shape (batch_size, channels, height, width) representing
      images normalized using the VGG-16 convention (BGR, ImageNet channel-wise
      mean-centered).
    score_threshold : float
      Minimum required score threshold (applied per class) for a detection to
      be considered. Set this higher for visualization to minimize extraneous
      boxes.
    anchor_map : torch.Tensor
      Map of anchors, shaped (height, width, num_anchors * 4). The last
      dimension contains the anchor boxes specified as a 4-tuple of
      (center_y, center_x, height, width), repeated for all anchors at that
      coordinate of the feature map. If this or anchor_valid_map is not
      provided, both will be computed here.
    anchor_valid_map : torch.Tensor
      Map indicating which anchors are valid (do not intersect image bounds),
      shaped (height, width). If this or anchor_map is not provided, both will
      be computed here.

    Returns
    -------
    Dict[int, np.ndarray]
      Scored boxes, (N, 5) tensor of box corners and class score,
      (y1, x1, y2, x2, score), indexed by class index.
    """
    assert image_data.shape[0] == 1, "Batch size must be 1"

    self.eval()

    # Forward inference
    proposals, classes, regressions = self(
      image_data = image_data,
      anchor_map = anchor_map,
      anchor_valid_map = anchor_valid_map
    )
    classes = classes.cpu().numpy()
    regressions = regressions.cpu().numpy()
 
    # Convert proposal boxes -> center point and size
    proposal_anchors = np.empty(proposals.shape)
    proposal_anchors[:,0] = 0.5 * (proposals[:,0] + proposals[:,2]) # center_y
    proposal_anchors[:,1] = 0.5 * (proposals[:,1] + proposals[:,3]) # center_x
    proposal_anchors[:,2:4] = proposals[:,2:4] - proposals[:,0:2]   # height, width

    # Separate out results per class: class_idx -> (y1, x1, y2, x2, score)
    boxes_and_scores_by_class_idx = {}
    for class_idx in range(1, classes.shape[1]):  # skip class 0 (background)
      # Get the regression parameters (ty, tx, th, tw) corresponding to this
      # class, for all proposals
      regression_idx = (class_idx - 0) * 4
      regression_params = regressions[:, (regression_idx + 0) : (regression_idx + 4)] # (N, 4)
      proposal_boxes_this_class = math_utils.convert_regressions_to_boxes(
        regressions = regression_params,
        anchors = proposal_anchors,
        regression_means = self._detector_regression_means,
        regression_stds = self._detector_regression_stds
      )

      # Clip to image boundaries
      proposal_boxes_this_class[:,0::2] = np.clip(proposal_boxes_this_class[:,0::2], 0, image_data.shape[2] - 1)  # clip y1 and y2 to [0,height)
      proposal_boxes_this_class[:,1::2] = np.clip(proposal_boxes_this_class[:,1::2], 0, image_data.shape[3] - 1)  # clip x1 and x2 to [0,width)

      # Get the scores for this class. The class scores are returned in
      # normalized categorical form. Each row corresponds to a class.
      scores_this_class = classes[:,class_idx]

      # Keep only those scoring high enough
      sufficiently_scoring_idxs = np.where(scores_this_class > score_threshold)[0]
      proposal_boxes_this_class = proposal_boxes_this_class[sufficiently_scoring_idxs]
      scores_this_class = scores_this_class[sufficiently_scoring_idxs]
      boxes_and_scores_by_class_idx[class_idx] = (proposal_boxes_this_class, scores_this_class)

    # Perform NMS per class
    scored_boxes_by_class_idx = {}
    for class_idx, (boxes, scores) in boxes_and_scores_by_class_idx.items():
      idxs = nms(
        boxes = t.from_numpy(boxes).cuda(),
        scores = t.from_numpy(scores).cuda(),
        iou_threshold = 0.3
      ).cpu().numpy()
      boxes = boxes[idxs]
      scores = np.expand_dims(scores[idxs], axis = 0) # (N,) -> (N,1)
      scored_boxes = np.hstack([ boxes, scores.T ])   # (N,5), with each row: (y1, x1, y2, x2, score)
      scored_boxes_by_class_idx[class_idx] = scored_boxes

    return scored_boxes_by_class_idx
