#
# TODO:
# -----
# - IoU threshold for prediction. Is it 0.3 as here or 0.5? Check the paper.
#

from dataclasses import dataclass
import numpy as np
import random
import torch as t
from torch import nn
from torchvision.ops import nms

from FasterRCNN import utils
from . import anchors
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

  def __init__(self, num_classes, rpn_minibatch_size = 256, proposal_batch_size = 128, allow_edge_proposals = True, dropout_probability = 0):
    super().__init__()

    # Constants
    self._num_classes = num_classes
    self._rpn_minibatch_size = rpn_minibatch_size
    self._proposal_batch_size = proposal_batch_size
    self._detector_regression_means = (0, 0, 0, 0)
    self._detector_regression_stds = (0.1, 0.1, 0.2, 0.2)

    # Network stages
    self._stage1_feature_extractor = vgg16.FeatureExtractor()
    self._stage2_region_proposal_network = rpn.RegionProposalNetwork(allow_edge_proposals = allow_edge_proposals)
    self._stage3_detector_network = detector.DetectorNetwork(num_classes = num_classes, dropout_probability = dropout_probability)

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
      - Box regression parameters (M, (num_classes - 1) * 4) from detector
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
    print(proposals)
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
    self.eval()
    assert image_data.shape[0] == 1, "Batch size must be 1"

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
      regression_idx = (class_idx - 1) * 4
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

  def train_step(self, optimizer, image_data, anchor_map, anchor_valid_map, gt_rpn_map, gt_rpn_object_indices, gt_rpn_background_indices, gt_boxes):
    """
    Performs one training step on a sample of data.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
      Optimizer.
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
    gt_rpn_map : torch.Tensor
      Ground truth RPN map of shape
      (batch_size, height, width, num_anchors, 6), where height and width are
      the feature map dimensions, not the input image dimensions. The final
      dimension contains:
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
    gt_rpn_object_indices : List[np.ndarray]
      For each image in the batch, a map of shape (N, 3) of indices (y, x, k)
      of all N object anchors in the RPN ground truth map.
    gt_rpn_background_indices : List[np.ndarray]
      For each image in the batch, a map of shape (M, 3) of indices of all M
      background anchors in the RPN ground truth map.
    gt_boxes : List[List[datasets.training_sample.Box]]
      For each image in the batch, a list of ground truth object boxes.

    Returns
    -------
    Loss, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray,
    np.ndarray
   
      1. Loss (a dataclass with class and regression losses for both the RPN
         and detector states).
      2. RPN objectness score map: (batch_size, height, width, num_anchors).
      3. RPN regressions map: (batch_size, height, width, num_anchors * 4),
         where the regressions are stored in the final dimension in
         parameterized form ((ty, tx, th, tw) for each anchor).
      4. Detected classes: (num_proposals, num_classes).
      5. Detected regressions: (num_proposals, 4*(num_classes-1)), stored in
         parameterized form relative to the proposal boxes (which are not
         returned). Note that class index 0 is the first non-background class.
      6. Ground truth classes: (num_proposals, num_classes), for the final
         detection stage.
      7. Ground truth regressions: (num_proposals, 4*(num_classes-1)), for the
         final detection stage.
    """
    self.train()

    # Clear accumulated gradient
    optimizer.zero_grad()

    # For now, we only support a batch size of 1
    assert image_data.shape[0] == 1, "Batch size must be 1"
    assert len(gt_rpn_map.shape) == 5 and gt_rpn_map.shape[0] == 1, "Batch size must be 1"
    assert len(gt_rpn_object_indices) == 1, "Batch size must be 1"
    assert len(gt_rpn_background_indices) == 1, "Batch size must be 1"
    assert len(gt_boxes) == 1, "Batch size must be 1"
    image_shape = image_data.shape[1:]

    # Stage 1: Extract features
    feature_map = self._stage1_feature_extractor(image_data = image_data)

    # Stage 2: Generate object proposals using RPN 
    rpn_score_map, rpn_regressions_map, proposals = self._stage2_region_proposal_network(
      feature_map = feature_map,
      image_shape = image_shape,  # each image in batch has identical shape: (num_channels, height, width)
      anchor_map = anchor_map,
      anchor_valid_map = anchor_valid_map,
      max_proposals_pre_nms = 12000,
      max_proposals_post_nms = 2000
    )

    # Sample random mini-batch of anchors (for RPN training)
    gt_rpn_minibatch_map = self._sample_rpn_minibatch(
      rpn_map = gt_rpn_map,
      object_indices = gt_rpn_object_indices,
      background_indices = gt_rpn_background_indices
    )

    # Assign labels to proposals and take random sample (for detector training)
    proposals, gt_classes, gt_regressions = self._label_proposals(
      proposals = proposals,
      gt_boxes = gt_boxes[0], # for now, batch size of 1
      min_background_iou_threshold = 0.0,
      min_object_iou_threshold = 0.5
    )
    proposals, gt_classes, gt_regressions = self._sample_proposals(
      proposals = proposals,
      gt_classes = gt_classes,
      gt_regressions = gt_regressions,
      max_proposals = self._proposal_batch_size,
      positive_fraction = 0.25
    )

    # Stage 3: Detector
    classes, regressions = self._stage3_detector_network(
      feature_map = feature_map,
      proposals = proposals 
    )

    # Compute losses
    rpn_class_loss = rpn.class_loss(predicted_scores = rpn_score_map, y_true = gt_rpn_minibatch_map)
    rpn_regression_loss = rpn.regression_loss(predicted_regressions = rpn_regressions_map, y_true = gt_rpn_minibatch_map)
    detector_class_loss = detector.class_loss(predicted_classes = classes, y_true = t.from_numpy(gt_classes).to("cuda"))
    detector_regression_loss = detector.regression_loss(predicted_regressions = regressions, y_true = t.from_numpy(gt_regressions).to("cuda"))
    loss = FasterRCNNModel.Loss(
      rpn_class = rpn_class_loss,
      rpn_regression = rpn_regression_loss,
      detector_class = detector_class_loss,
      detector_regression = detector_regression_loss,
      total = rpn_class_loss + rpn_regression_loss + detector_class_loss + detector_regression_loss
    )

    # Backprop
    loss.total.backward()

    # Optimizer step
    optimizer.step()

    # Return losses and data useful for computing statistics
    return loss, rpn_score_map, rpn_regressions_map, classes, regressions, gt_classes, gt_regressions
  
  def _sample_rpn_minibatch(self, rpn_map, object_indices, background_indices):
    """
    Selects anchors for training and produces a copy of the RPN ground truth
    map with only those anchors marked as trainable.

    Parameters
    ----------
    rpn_map : np.ndarray
      RPN ground truth map of shape
      (batch_size, height, width, num_anchors, 6).
    object_indices : List[np.ndarray]
      For each image in the batch, a map of shape (N, 3) of indices (y, x, k)
      of all N object anchors in the RPN ground truth map.
    background_indices : List[np.ndarray]
      For each image in the batch, a map of shape (M, 3) of indices of all M
      background anchors in the RPN ground truth map.

    Returns
    -------
    np.ndarray
      A copy of the RPN ground truth map with index 0 of the last dimension
      recomputed to include only anchors in the minibatch.
    """
    assert rpn_map.shape[0] == 1, "Batch size must be 1"
    assert len(object_indices) == 1, "Batch size must be 1"
    assert len(background_indices) == 1, "Batch size must be 1"
    positive_anchors = object_indices[0]
    negative_anchors = background_indices[0]
    assert len(positive_anchors) + len(negative_anchors) >= self._rpn_minibatch_size, "Image has insufficient anchors for RPN minibatch size of %d" % self._rpn_minibatch_size
    assert len(positive_anchors) > 0, "Image does not have any positive anchors"
    assert self._rpn_minibatch_size % 2 == 0, "RPN minibatch size must be evenly divisible"

    # Sample, producing indices into the index maps
    num_positive_anchors = len(positive_anchors)
    num_negative_anchors = len(negative_anchors)
    num_positive_samples = min(self._rpn_minibatch_size // 2, num_positive_anchors) # up to half the samples should be positive, if possible
    num_negative_samples = self._rpn_minibatch_size - num_positive_samples          # the rest should be negative
    #positive_anchor_idxs = random.sample(range(num_positive_anchors), num_positive_samples)
    #negative_anchor_idxs = random.sample(range(num_negative_anchors), num_negative_samples)
    positive_anchor_idxs = np.arange(num_positive_anchors)
    negative_anchor_idxs = np.arange(num_negative_anchors)
    
    # Construct index expressions into RPN map
    positive_anchors = positive_anchors[positive_anchor_idxs]
    negative_anchors = negative_anchors[negative_anchor_idxs]
    trainable_anchors = np.concatenate([ positive_anchors, negative_anchors ])
    batch_idxs = np.zeros(len(trainable_anchors))
    trainable_idxs = (batch_idxs, trainable_anchors[:,0], trainable_anchors[:,1], trainable_anchors[:,2], 0)

    # Create a copy of the RPN map with samples set as trainable
    rpn_minibatch_map = rpn_map.clone()
    rpn_minibatch_map[:,:,:,:,0] = 0
    rpn_minibatch_map[trainable_idxs] = 1

    return rpn_minibatch_map

  def _label_proposals(self, proposals, gt_boxes, min_background_iou_threshold, min_object_iou_threshold):
    """
    Determines which proposals generated by the RPN stage overlap with ground
    truth boxes and creates ground truth labels for the subsequent detector
    stage.

    Parameters
    ----------
    proposals : np.ndarray
      Proposal corners, shaped (N, 4).
    gt_boxes : List[datasets.training_sample.Box]
      Ground truth object boxes.
    min_background_iou_threshold : float
      Minimum IoU threshold with ground truth boxes below which proposals are
      ignored entirely. Proposals with an IoU threshold in the range
      [min_background_iou_threshold, min_object_iou_threshold) are labeled as
      background. This value can be greater than 0, which has the effect of
      selecting more difficult background examples that have some degree of
      overlap with ground truth boxes.
    min_object_iou_threshold : float
      Minimum IoU threshold for a proposal to be labeled as an object.

    Returns
    -------
    np.ndarray, np.ndarray, np.ndarray
      Proposals, (N, 4), labeled as either objects or background (depending on
      IoU thresholds, some proposals can end up as neither and are excluded
      here); one-hot encoded class labels, (N, num_classes), for each proposal;
      and regression targets, (N, 2, (num_classes - 1) * 4), for each proposal.
      Regression target values are present at locations [:,1,:] and consist of
      (ty, tx, th, tw) for the class that the box corresponds to. The entries
      for all other classes and the background classes should be ignored. A
      mask is written to locations [:,0,:]. For each proposal assigned a non-
      background class, there will be 4 consecutive elements marked with 1
      indicating the corresponding regression target values are to be used.
      There are no regression targets for background proposals and the mask is
      entirely 0 for those proposals.
    """
    assert min_background_iou_threshold < min_object_iou_threshold, "Object threshold must be greater than background threshold"

    # Convert ground truth box corners to (M,4) tensor and class indices to (M,)
    gt_box_corners = np.array([ box.corners for box in gt_boxes ]).astype(np.float32)
    gt_box_class_idxs = np.array([ box.class_index for box in gt_boxes ]).astype(np.int32)

    # Let's be crafty and create some fake proposals that match the ground
    # truth boxes exactly. This isn't strictly necessary and the model should
    # work without it but it will help training and will ensure that there are
    # always some positive examples to train on. 
    proposals = np.vstack([ proposals, gt_box_corners ])

    # Compute IoU between each proposal (N,4) and each ground truth box (M,4)
    # -> (N, M)
    ious = math_utils.intersection_over_union(boxes1 = proposals, boxes2 = gt_box_corners)

    # Find the best IoU for each proposal, the class of the ground truth box
    # associated with it, and the box corners
    best_ious = np.max(ious, axis = 1)              # (N,) of maximum IoUs for each of the N proposals
    box_idxs = np.argmax(ious, axis = 1)            # (N,) of ground truth box index for each proposal
    gt_box_class_idxs = gt_box_class_idxs[box_idxs] # (N,) of class indices of highest-IoU box for each proposal
    gt_box_corners = gt_box_corners[box_idxs]       # (N,4) of box corners of highest-IoU box for each proposal
 
    # Remove all proposals whose best IoU is less than the minimum threshold
    # for a negative (background) sample. We also check for IoUs > 0 because
    # due to earlier clipping, we may get invalid 0-area proposals.
    idxs = np.where((best_ious >= min_background_iou_threshold))[0]  # keep proposals w/ sufficiently high IoU
    proposals = proposals[idxs]
    best_ious = best_ious[idxs]
    gt_box_class_idxs = gt_box_class_idxs[idxs]
    gt_box_corners = gt_box_corners[idxs]

    # IoUs less than min_object_iou_threshold will be labeled as background
    gt_box_class_idxs[best_ious < min_object_iou_threshold] = 0
    
    # One-hot encode class labels
    num_proposals = proposals.shape[0]
    gt_classes = np.zeros((num_proposals, self._num_classes)).astype(np.float32)  # (N,num_classes)
    gt_classes[ np.arange(num_proposals), gt_box_class_idxs ] = 1.0

    # Convert proposals and ground truth boxes into "anchor" format (center
    # points and side lengths). For the detector stage, the proposals serve as
    # the anchors relative to which the final box predictions will be 
    # regressed.
    proposal_centers = 0.5 * (proposals[:,0:2] + proposals[:,2:4])          # center_y, center_x
    proposal_sides = proposals[:,2:4] - proposals[:,0:2]                    # height, width
    gt_box_centers = 0.5 * (gt_box_corners[:,0:2] + gt_box_corners[:,2:4])  # center_y, center_x
    gt_box_sides = gt_box_corners[:,2:4] - gt_box_corners[:,0:2]            # height, width

    # Compute regression targets (ty, tx, th, tw) for each proposal based on
    # the best box selected
    regression_targets = np.empty((num_proposals, 4))                       # (N,4)
    regression_targets[:,0:2] = (gt_box_centers - proposal_centers) / proposal_sides  # ty = (gt_center_y - proposal_center_y) / proposal_height, tx = (gt_center_x - proposal_center_x) / proposal_width
    regression_targets[:,2:4] = np.log(gt_box_sides / proposal_sides)                 # th = log(gt_height / proposal_height), tw = (gt_width / proposal_width)
    regression_targets[:,:] -= self._detector_regression_means              # mean adjustment
    regression_targets[:,:] /= self._detector_regression_stds               # standard deviation scaling

    # Convert regression targets into a map of shape (N,2,4*(C-1)) where C is
    # the number of classes and [:,0,:] specifies a mask for the corresponding
    # target components at [:,1,:]. Targets are ordered (ty, tx, th, tw).
    # Background class 0 is not present at all.
    gt_regressions = np.zeros((num_proposals, 2, 4 * (self._num_classes - 1))).astype(np.float32)
    gt_regressions[:,0,:] = np.repeat(gt_classes, repeats = 4, axis = 1)[:,4:]        # create masks using interleaved repetition, remembering to ignore class 0
    gt_regressions[:,1,:] = np.tile(regression_targets, reps = self._num_classes - 1) # populate regression targets with straightforward repetition (only those columns corresponding to class are masked on)

    np.set_printoptions(threshold=np.inf)
    print("proposals=", proposals)
    print("gt_classes=", gt_classes)
    print("gt_regressions=", gt_regressions)

    return proposals, gt_classes, gt_regressions

  def _sample_proposals(self, proposals, gt_classes, gt_regressions, max_proposals, positive_fraction):
    if max_proposals <= 0:
      return proposals, gt_classes, gt_regressions
  
    # Get positive and negative (background) proposals
    class_indices = np.argmax(gt_classes, axis = 1)  # (N,num_classes) -> (N,), where each element is the class index (highest score from its row)
    positive_indices = np.argwhere(class_indices > 0)[:,0]
    negative_indices = np.argwhere(class_indices <= 0)[:,0]
    num_positive_proposals = len(positive_indices)
    num_negative_proposals = len(negative_indices)
    
    # Select positive and negative samples, if there are enough. Note that the
    # number of positive samples can be either the positive fraction of the
    # *actual* number of proposals *or* the *desired* number (max_proposals).
    # In practice, these yield virtually identical results but the latter
    # method will yield slightly more positive samples in the rare cases when 
    # the number of proposals is below the desired number. Here, we use the
    # former method but others, such as Yun Chen, use the latter. To implement
    # it, replace num_samples with max_proposals in the line that computes
    # num_positive_samples. I am not sure what the original FasterRCNN
    # implementation does.
    num_samples = min(max_proposals, len(class_indices))
    num_positive_samples = min(round(num_samples * positive_fraction), num_positive_proposals)
    num_negative_samples = min(num_samples - num_positive_samples, num_negative_proposals)
  
    # Do we have enough?
    if num_positive_samples <= 0 or num_negative_samples <= 0:
      return proposals[[]], gt_classes[[]], gt_regressions[[]]  # return 0-length tensors
  
    # Sample randomly
    positive_sample_indices = np.random.choice(positive_indices, size = num_positive_samples, replace = False)
    negative_sample_indices = np.random.choice(negative_indices, size = num_negative_samples, replace = False)
    indices = np.concatenate([ positive_sample_indices, negative_sample_indices ])
  
    # Return
    return proposals[indices], gt_classes[indices], gt_regressions[indices]
