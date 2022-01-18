#
# Faster R-CNN in PyTorch and TensorFlow 2 w/ Keras
# tf2/FasterRCNN/models/rpn.py
# Copyright 2021-2022 Bart Trzynadlowski
# 
# TensorFlow/Keras implementation of the RPN (region proposal network) stage of
# Faster R-CNN. Given a feature map (i.e., the output of the VGG-16
# convolutional layers), generates objectness scores for each anchor box, and
# boxes in the form of modifications to anchor center points and dimensions.
#
# Unlike the original Faster R-CNN implementation (and many subsequent re-
# implementations), which used two outputs per anchor (object and background)
# and a softmax activation, this implementation uses only a single output and
# sigmoid activation, which is simpler but equivalent. A value of < 0.5 is
# background and >= 0.5 is an object.
#
# The RPN class and box delta regression losses are defined here.
#

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import models
from tensorflow.keras.layers import Conv2D
from tensorflow.keras import backend as K

from . import math_utils


class RegionProposalNetwork(tf.keras.Model):
  def __init__(self, max_proposals_pre_nms_train, max_proposals_post_nms_train, max_proposals_pre_nms_infer, max_proposals_post_nms_infer, l2 = 0, allow_edge_proposals = False):
    super().__init__()

    self._max_proposals_pre_nms_train = max_proposals_pre_nms_train
    self._max_proposals_post_nms_train = max_proposals_post_nms_train
    self._max_proposals_pre_nms_infer = max_proposals_pre_nms_infer
    self._max_proposals_post_nms_infer = max_proposals_post_nms_infer
    self._allow_edge_proposals = allow_edge_proposals

    regularizer = tf.keras.regularizers.l2(l2)
    initial_weights = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.01, seed = None)

    anchors_per_location = 9

    # 3x3 convolution over input map producing 512-d result at each output. The center of each output is an anchor point (k anchors at each point).
    self._rpn_conv1 = Conv2D(name = "rpn_conv1", kernel_size = (3,3), strides = 1, filters = 512, padding = "same", activation = "relu", kernel_initializer = initial_weights, kernel_regularizer = regularizer)

    # Classification layer: predicts whether there is an object at the anchor or not. We use a sigmoid function, where > 0.5 is indicates a positive result.
    self._rpn_class = Conv2D(name = "rpn_class", kernel_size = (1,1), strides = 1, filters = anchors_per_location, padding = "same", activation = "sigmoid", kernel_initializer = initial_weights)

    # Box delta regression
    self._rpn_boxes = Conv2D(name = "rpn_boxes", kernel_size = (1,1), strides = 1, filters = 4 * anchors_per_location, padding = "same", activation = None, kernel_initializer = initial_weights)

  def __call__(self, inputs, training):
    # Unpack inputs
    input_image = inputs[0]
    feature_map = inputs[1]
    anchor_map = inputs[2]
    anchor_valid_map = inputs[3]
    assert len(feature_map.shape) == 4

    # Proposal sampling depends on whether we are training or not
    if training:
      max_proposals_pre_nms = self._max_proposals_pre_nms_train
      max_proposals_post_nms = self._max_proposals_post_nms_train
    else:
      max_proposals_pre_nms = self._max_proposals_pre_nms_infer
      max_proposals_post_nms = self._max_proposals_post_nms_infer

    # Pass through network
    y = self._rpn_conv1(feature_map)
    scores = self._rpn_class(y)
    box_delta_regressions = self._rpn_boxes(y)

    # Extract valid
    anchors, objectness_scores, box_deltas = self._extract_valid(
      anchor_map = anchor_map,
      anchor_valid_map = anchor_valid_map,
      objectness_score_map = scores,
      box_delta_map = box_delta_regressions,
      allow_edge_proposals = self._allow_edge_proposals
    )

    # Convert regressions to box corners
    proposals = math_utils.tf_convert_deltas_to_boxes(
      box_deltas = box_deltas,
      anchors = anchors,
      box_delta_means = [ 0.0, 0.0, 0.0, 0.0 ],
      box_delta_stds = [ 1.0, 1.0, 1.0, 1.0 ]
    )
    
    # Keep only the top-N scores. Note that we do not care whether the
    # proposals were labeled as objects (score > 0.5) and peform a simple
    # ranking among all of them. Restricting them has a strong adverse impact
    # on training performance.
    sorted_indices = tf.argsort(objectness_scores)                  # sort in ascending order of objectness score
    sorted_indices = sorted_indices[::-1]                           # descending order of score
    proposals = tf.gather(proposals, indices = sorted_indices)[0:max_proposals_pre_nms] # grab the top-N best proposals
    objectness_scores = tf.gather(objectness_scores, indices = sorted_indices)[0:max_proposals_pre_nms] # corresponding scores

    # Clip to image boundaries
    image_height = tf.cast(tf.shape(input_image)[1], dtype = tf.float32)  # height in pixels (shape of image is (batches,height,width,channels))
    image_width = tf.cast(tf.shape(input_image)[2], dtype = tf.float32)   # width in pixels
    proposals_top_left = tf.maximum(proposals[:,0:2], 0.0)
    proposals_y2 = tf.reshape(tf.minimum(proposals[:,2], image_height), shape = (-1, 1))  # slice operation produces [N,], reshape to [N,1]
    proposals_x2 = tf.reshape(tf.minimum(proposals[:,3], image_width), shape = (-1, 1))
    proposals = tf.concat([ proposals_top_left, proposals_y2, proposals_x2 ], axis = 1) # [N,4] proposal tensor

    # Remove anything less than 16 pixels on a side
    height = proposals[:,2] - proposals[:,0]
    width = proposals[:,3] - proposals[:,1]
    idxs = tf.where((height >= 16) & (width >= 16))
    proposals = tf.gather_nd(proposals, indices = idxs)
    objectness_scores = tf.gather_nd(objectness_scores, indices = idxs)

    # Perform NMS
    idxs = tf.image.non_max_suppression(
      boxes = proposals,
      scores = objectness_scores,
      max_output_size = max_proposals_post_nms,
      iou_threshold = 0.7
    )
    proposals = tf.gather(proposals, indices = idxs)

    return [ scores, box_delta_regressions, proposals ] 

  def _extract_valid(self, anchor_map, anchor_valid_map, objectness_score_map, box_delta_map, allow_edge_proposals):
    # anchor_valid_map shape is (batch,height,width,num_anchors)
    height = tf.shape(anchor_valid_map)[1]
    width = tf.shape(anchor_valid_map)[2]
    num_anchors = tf.shape(anchor_valid_map)[3]
  
    anchors = tf.reshape(anchor_map, shape = (height * width * num_anchors, 4))             # [N,4], all anchors 
    anchors_valid = tf.reshape(anchor_valid_map, shape = (height * width * num_anchors, 1)) # [N,1], whether anchors are valid (i.e., do not cross image boundaries)
    scores = tf.reshape(objectness_score_map, shape = (height * width * num_anchors, 1))    # [N,1], predicted objectness scores
    box_deltas = tf.reshape(box_delta_map, shape = (height * width * num_anchors, 4))       # [N,4], predicted box delta regression targets
    
    anchors_valid = tf.squeeze(anchors_valid)                                               # [N,]
    scores = tf.squeeze(scores)                                                             # [N,]
  
    if allow_edge_proposals:
      # Use all proposals
      return anchors, scores, box_deltas
    else:
      # Filter out those proposals generated at invalid anchors. Invalid
      # anchors are really just those that cross image boundaries and, counter-
      # intuitively, given that the Faster R-CNN paper (Section 3.3) says that
      # these anchors are ignored during loss calculation, they should in fact
      # be included when generating proposals. Good performance requires
      # evaluating lots of proposals, so even if cross-boundary anchors do not
      # contribute to RPN loss, they can still feed samples into the detector
      # stage. It is therefore not recommended to exclude edge proposals but
      # the option exists here for educational purposes.
      idxs = tf.where(anchors_valid > 0)
      return tf.gather_nd(anchors, indices = idxs), tf.gather_nd(scores, indices = idxs), tf.gather_nd(box_deltas, indices = idxs)

  @staticmethod
  def class_loss(y_predicted, gt_rpn_map):
    """
    Computes RPN class loss.
  
    Parameters
    ----------
    y_predicted : tf.Tensor
      A tensor of shape (batch_size, height, width, num_anchors) containing
      objectness scores (0 = background, 1 = object).
    gt_rpn_map : tf.Tensor
      Ground truth tensor of shape (batch_size, height, width, num_anchors, 6).
  
    Returns
    -------
    tf.Tensor
      Scalar loss.
    """
  
    # y_true_class: (batch_size, height, width, num_anchors), same as predicted_scores
    y_true_class = tf.reshape(gt_rpn_map[:,:,:,:,1], shape = tf.shape(y_predicted))
    y_predicted_class = y_predicted
    
    # y_mask: y_true[:,:,:,0] is 1.0 for anchors included in the mini-batch
    y_mask = tf.reshape(gt_rpn_map[:,:,:,:,0], shape = tf.shape(y_predicted_class))
  
    # Compute how many anchors are actually used in the mini-batch (e.g.,
    # typically 256)
    N_cls = tf.cast(tf.math.count_nonzero(y_mask), dtype = tf.float32) + K.epsilon()
  
    # Compute element-wise loss for all anchors
    loss_all_anchors = K.binary_crossentropy(y_true_class, y_predicted_class)
    
    # Zero out the ones which should not have been included
    relevant_loss_terms = y_mask * loss_all_anchors
  
    # Sum the total loss and normalize by the number of anchors used
    return K.sum(relevant_loss_terms) / N_cls
  
  @staticmethod
  def regression_loss(y_predicted, gt_rpn_map):
    """
    Computes RPN box delta regression loss.
  
    Parameters
    ----------
    y_predicted : tf.Tensor
      A tensor of shape (batch_size, height, width, num_anchors * 4) containing
      RoI box delta regressions for each anchor, stored as: ty, tx, th, tw.
    gt_rpn_map : tf.Tensor
      Ground truth tensor of shape (batch_size, height, width, num_anchors, 6).
  
    Returns
    -------
    tf.Tensor
      Scalar loss.
    """
  
    scale_factor = 1.0  # hyper-parameter that controls magnitude of regression loss and is chosen to make regression term comparable to class term
    sigma = 3.0         # see: https://github.com/rbgirshick/py-faster-rcnn/issues/89
    sigma_squared = sigma * sigma
  
    y_predicted_regression = y_predicted
    y_true_regression = tf.reshape(gt_rpn_map[:,:,:,:,2:6], shape = tf.shape(y_predicted_regression))
  
    # Include only anchors that are used in the mini-batch and which correspond
    # to objects (positive samples)
    y_included = tf.reshape(gt_rpn_map[:,:,:,:,0], shape = tf.shape(gt_rpn_map)[0:4]) # trainable anchors map: (batch_size, height, width, num_anchors)
    y_positive = tf.reshape(gt_rpn_map[:,:,:,:,1], shape = tf.shape(gt_rpn_map)[0:4]) # positive anchors
    y_mask = y_included * y_positive
  
    # y_mask is of the wrong shape. We have one value per (y,x,k) position but in
    # fact need to have 4 values (one for each of the regression variables). For
    # example, y_predicted might be (1,37,50,36) and y_mask will be (1,37,50,9).
    # We need to repeat the last dimension 4 times.
    y_mask = tf.repeat(y_mask, repeats = 4, axis = 3)
  
    # The paper normalizes by dividing by a quantity called N_reg, which is equal
    # to the total number of anchors (~2400) and then multiplying by lambda=10.
    # This does not make sense to me because we are summing over a mini-batch at
    # most, so we use N_cls here. I might be misunderstanding what is going on
    # but 10/2400 = 1/240 which is pretty close to 1/256 and the paper mentions
    # that training is relatively insensitve to choice of normalization.
    N_cls = tf.cast(tf.math.count_nonzero(y_included), dtype = tf.float32) + K.epsilon()
  
    # Compute element-wise loss using robust L1 function for all 4 regression
    # components
    x = y_true_regression - y_predicted_regression
    x_abs = tf.math.abs(x)
    is_negative_branch = tf.stop_gradient(tf.cast(tf.less(x_abs, 1.0 / sigma_squared), dtype = tf.float32))
    R_negative_branch = 0.5 * x * x * sigma_squared
    R_positive_branch = x_abs - 0.5 / sigma_squared
    loss_all_anchors = is_negative_branch * R_negative_branch + (1.0 - is_negative_branch) * R_positive_branch
  
    # Zero out the ones which should not have been included
    relevant_loss_terms = y_mask * loss_all_anchors
    return scale_factor * K.sum(relevant_loss_terms) / N_cls
