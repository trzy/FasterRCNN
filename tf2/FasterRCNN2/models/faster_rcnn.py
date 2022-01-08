import sys
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Lambda

from . import vgg16
from . import rpn
from . import detector


def faster_rcnn_model(mode, num_classes, allow_edge_proposals, custom_roi_pool, detector_class_activations, l2 = 0):
  assert mode == "train" or mode == "infer"
  if mode == "train":
    return _training_model(num_classes, allow_edge_proposals, custom_roi_pool, detector_class_activations, l2 = l2)
  else:
    return _inference_model(num_classes, allow_edge_proposals, custom_roi_pool, detector_class_activations)

def _inference_model(num_classes, allow_edge_proposals, custom_roi_pool, detector_class_activations):
  image_shape_map = Input(shape = (3,), name = "image_shape_map")                         # holds shape of image: height, width, channels
  num_anchors = 9
  anchor_map = Input(shape = (None, None, num_anchors * 4), name = "anchor_map")          # (height, width, k*4)
  anchor_valid_map = Input(shape = (None, None, num_anchors), name = "anchor_valid_map")  # (height, width, k)
  
  #
  # Define model
  #
  
  # Stage 1: Extract features 
  stage1_feature_extractor_model = vgg16.conv_layers()

  # Stage 2: Generate object proposals using RPN 
  rpn_score_output, rpn_regression_output, proposals_output = rpn.layers(
    input_map = stage1_feature_extractor_model.outputs[0],
    image_shape_map = image_shape_map,
    anchor_map = anchor_map,
    anchor_valid_map = anchor_valid_map,
    max_proposals_pre_nms = 6000, # test time values 
    max_proposals_post_nms = 300,
    allow_edge_proposals = allow_edge_proposals 
  )
  import sys
  def do_log(x):
    tf.print(x, output_stream = sys.stdout, summarize = -1)
    return x
  proposals_output = Lambda(do_log)(proposals_output)
 
  # Stage 3: Detector
  detector_class_output, detector_regression_output = detector.layers(
    image_shape = tf.shape(stage1_feature_extractor_model.input),
    feature_map = stage1_feature_extractor_model.outputs[0],
    proposals = proposals_output,
    num_classes = num_classes,
    custom_roi_pool = custom_roi_pool,
    detector_class_activations = detector_class_activations,
    l2 = 0
  )

  # Build model
  model = Model(
    # Inputs
    [
      stage1_feature_extractor_model.input,
      image_shape_map,
      anchor_map,
      anchor_valid_map
    ],
    # Outputs
    [
      rpn_score_output,
      rpn_regression_output,
      detector_class_output,
      detector_regression_output,
      proposals_output
    ]
  )

  return model

def _training_model(num_classes, allow_edge_proposals, custom_roi_pool, detector_class_activations, l2 = 0):
  image_shape_map = Input(shape = (3,), name = "image_shape_map")                         # holds shape of image: height, width, channels
  num_anchors = 9
  anchor_map = Input(shape = (None, None, num_anchors * 4), name = "anchor_map")          # (height, width, k*4)
  anchor_valid_map = Input(shape = (None, None, num_anchors), name = "anchor_valid_map")  # (height, width, k)
  gt_rpn_map = Input(shape = (None, None, num_anchors, 6), name = "gt_rpn_map")           # (height, width, k, 6)
  gt_box_class_idxs_map = Input(shape = (None,), dtype = tf.int32, name = "gt_box_class_idxs")  # (num_gt_boxes,)
  gt_box_corners_map = Input(shape = (None,4), name = "gt_box_corners")                   # (num_gt_boxes,4)

  #
  # Define model
  #

  # Stage 1: Extract features 
  stage1_feature_extractor_model = vgg16.conv_layers(l2 = l2)

  # Stage 2: Generate object proposals using RPN 
  rpn_score_output, rpn_regression_output, proposals_output = rpn.layers(
    input_map = stage1_feature_extractor_model.outputs[0],
    image_shape_map = image_shape_map,
    anchor_map = anchor_map,
    anchor_valid_map = anchor_valid_map,
    max_proposals_pre_nms = 12000,
    max_proposals_post_nms = 2000,
    allow_edge_proposals = allow_edge_proposals,
    l2 = l2
  )

  # Assign labels to proposals and take random sample (for detector training)
  proposals, gt_classes, gt_regressions = _label_proposals(
    proposals = proposals_output,
    gt_box_class_idxs = gt_box_class_idxs_map[0], # for now, batch size of 1
    gt_box_corners = gt_box_corners_map[0],
    num_classes = num_classes,
    min_background_iou_threshold = 0.0,
    min_object_iou_threshold = 0.5
  )
  proposals, gt_classes, gt_regressions = _sample_proposals(
    proposals = proposals,
    gt_classes = gt_classes,
    gt_regressions = gt_regressions,
    max_proposals = 128,
    positive_fraction = 0.25
  )
  gt_classes = tf.expand_dims(gt_classes, axis = 0)           # (N,num_classes) -> (1,N,num_classes) (as expected by loss function)
  gt_regressions = tf.expand_dims(gt_regressions, axis = 0)   # (N,2,(num_classes-1)*4) -> (1,N,2,(num_classes-1)*4)

  # Stage 3: Detector
  detector_class_output, detector_regression_output = detector.layers(
    image_shape = tf.shape(stage1_feature_extractor_model.input),
    feature_map = stage1_feature_extractor_model.outputs[0],
    proposals = proposals,
    num_classes = num_classes,
    custom_roi_pool = custom_roi_pool,
    detector_class_activations = detector_class_activations,
    l2 = l2
  )

  # Losses
  rpn_class_loss = rpn.class_loss(y_predicted = rpn_score_output, gt_rpn_map = gt_rpn_map)
  rpn_regression_loss = rpn.regression_loss(y_predicted = rpn_regression_output, gt_rpn_map = gt_rpn_map)
  detector_class_loss = detector.class_loss(y_predicted = detector_class_output, y_true = gt_classes, from_logits = not detector_class_activations)
  detector_regression_loss = detector.regression_loss(y_predicted = detector_regression_output, y_true = gt_regressions)

  # Build model
  model = Model(
    # Inputs
    [
      stage1_feature_extractor_model.input,
      image_shape_map,
      anchor_map,
      anchor_valid_map,
      gt_rpn_map,
      gt_box_class_idxs_map,
      gt_box_corners_map
    ],
    # Outputs
    [
      rpn_score_output,
      rpn_regression_output,
      detector_class_output,
      detector_regression_output,
      proposals,
      rpn_class_loss,
      rpn_regression_loss,
      detector_class_loss,
      detector_regression_loss
    ]
  )
  model.add_loss(rpn_class_loss)
  model.add_loss(rpn_regression_loss)
  model.add_loss(detector_class_loss)
  model.add_loss(detector_regression_loss)
  model.add_metric(rpn_class_loss, name = "rpn_class_loss")
  model.add_metric(rpn_regression_loss, name = "rpn_regression_loss")
  model.add_metric(detector_class_loss, name = "detector_class_loss")
  model.add_metric(detector_regression_loss, name = "detector_regression_loss")

  return model

def _intersection_over_union(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
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

def _label_proposals(proposals, gt_box_class_idxs, gt_box_corners, num_classes, min_background_iou_threshold, min_object_iou_threshold):
    # Let's be crafty and create some fake proposals that match the ground
    # truth boxes exactly. This isn't strictly necessary and the model should
    # work without it but it will help training and will ensure that there are
    # always some positive examples to train on. 
    proposals = tf.concat([ proposals, gt_box_corners ], axis = 0)

    # Compute IoU between each proposal (N,4) and each ground truth box (M,4)
    # -> (N, M)
    ious = _intersection_over_union(boxes1 = proposals, boxes2 = gt_box_corners)

    # Find the best IoU for each proposal, the class of the ground truth box
    # associated with it, and the box corners
    best_ious = tf.math.reduce_max(ious, axis = 1)  # (N,) of maximum IoUs for each of the N proposals
    box_idxs = tf.math.argmax(ious, axis = 1)       # (N,) of ground truth box index for each proposal
    gt_box_class_idxs = tf.gather(gt_box_class_idxs, indices = box_idxs)  # (N,) of class indices of highest-IoU box for each proposal
    gt_box_corners = tf.gather(gt_box_corners, indices = box_idxs)     # (N,4) of box corners of highest-IoU box for each proposal
  
    """
    def do_log1(x):
      tf.print("best_ious=", x, output_stream = "file:///projects/FasterRCNN/tf2/out.txt", summarize = -1)
      return x
    best_ious = Lambda(do_log1)(best_ious)
    def do_log2(x):
      tf.print("best_ious=", x, output_stream = "file:///projects/FasterRCNN/tf2/out.txt", summarize = -1)
      return x
    box_idxs = Lambda(do_log2)(box_idxs)
    """

    # Remove all proposals whose best IoU is less than the minimum threshold
    # for a negative (background) sample. We also check for IoUs > 0 because
    # due to earlier clipping, we may get invalid 0-area proposals.
    idxs = tf.where(best_ious >= min_background_iou_threshold)  # keep proposals w/ sufficiently high IoU
    proposals = tf.gather_nd(proposals, indices = idxs)
    best_ious = tf.gather_nd(best_ious, indices = idxs)
    gt_box_class_idxs = tf.gather_nd(gt_box_class_idxs, indices = idxs)
    gt_box_corners = tf.gather_nd(gt_box_corners, indices = idxs)
    
    """
    def do_log3(x):
      tf.print("best_ious_filtered=", x, output_stream = "file:///projects/FasterRCNN/tf2/out.txt", summarize = -1)
      return x
    best_ious = Lambda(do_log3)(best_ious)
    def do_log4(x):
      tf.print("gt_box_class_idxs=", x, output_stream = "file:///projects/FasterRCNN/tf2/out.txt", summarize = -1)
      return x
    gt_box_class_idxs = Lambda(do_log4)(gt_box_class_idxs)
    """

    # IoUs less than min_object_iou_threshold will be labeled as background
    retain_mask = tf.cast(best_ious >= min_object_iou_threshold, dtype = gt_box_class_idxs.dtype) # (N,), with 0 wherever best_iou < threshold, else 1
    """
    def do_log5(x):
      tf.print("retain_mask=", x, output_stream = "file:///projects/FasterRCNN/tf2/out.txt", summarize = -1)
      return x
    retain_mask = Lambda(do_log5)(retain_mask)
    """
    gt_box_class_idxs = gt_box_class_idxs * retain_mask

    # One-hot encode class labels
    gt_classes = tf.one_hot(indices = gt_box_class_idxs, depth = num_classes) # (N,num_classes)

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
    detector_regression_means = tf.constant([0, 0, 0, 0], dtype = tf.float32)
    detector_regression_stds = tf.constant([0.1, 0.1, 0.2, 0.2], dtype = tf.float32)
    tyx = (gt_box_centers - proposal_centers) / proposal_sides  # ty = (gt_center_y - proposal_center_y) / proposal_height, tx = (gt_center_x - proposal_center_x) / proposal_width
    thw = tf.math.log(gt_box_sides / proposal_sides)        # th = log(gt_height / proposal_height), tw = (gt_width / proposal_width)
    regression_targets = tf.concat([ tyx, thw ], axis = 1)  # (N,4) regression targets tensor
    regression_targets = (regression_targets - detector_regression_means) / detector_regression_stds  # mean and standard deviation adjustment
 
    # Convert regression targets into a map of shape (N,2,4*(C-1)) where C is
    # the number of classes and [:,0,:] specifies a mask for the corresponding
    # target components at [:,1,:]. Targets are ordered (ty, tx, th, tw).
    # Background class 0 is not present at all.
    gt_regressions_mask = tf.repeat(gt_classes, repeats = 4, axis = 1)[:,4:]              # create masks using interleaved repetition, remembering to discard class 0
    gt_regressions_values = tf.tile(regression_targets, multiples = [1, num_classes - 1]) # populate regression targets with straightforward repetition of each row (only those columns corresponding to class will be masked on)
    gt_regressions_mask = tf.expand_dims(gt_regressions_mask, axis = 0)     # (N,4*(C-1)) -> (1,N,4*(C-1))
    gt_regressions_values = tf.expand_dims(gt_regressions_values, axis = 0) # (N,4*(C-1)) -> (1,N,4*(C-1))
    gt_regressions = tf.concat([ gt_regressions_mask, gt_regressions_values ], axis = 0)  # (2,N,4*(C-1))
    gt_regressions = tf.transpose(gt_regressions, perm = [ 1, 0, 2])        # (N,2,4*(C-1)) 

    """
    def do_log6(x):
      tf.print("proposals=", x, output_stream = "file:///projects/FasterRCNN/tf2/out.txt", summarize = -1)
      return x
    proposals = Lambda(do_log6)(proposals)
    """
    def do_log6(x):
      tf.print("proposals=", x, output_stream = sys.stdout, summarize = -1)
      return x
    proposals = Lambda(do_log6)(proposals)
    def do_log7(x):
      tf.print("gt_classes=", x, output_stream = sys.stdout, summarize = -1)
      return x
    gt_classes = Lambda(do_log7)(gt_classes)
    def do_log8(x):
      tf.print("gt_regressions=", x, output_stream = sys.stdout, summarize = -1)
      return x
    gt_regressions = Lambda(do_log8)(gt_regressions)
    return proposals, gt_classes, gt_regressions

def _sample_proposals(proposals, gt_classes, gt_regressions, max_proposals, positive_fraction):
  if max_proposals <= 0:
    return proposals, gt_classes, gt_regressions

  # Get positive and negative (background) proposals
  class_indices = tf.argmax(gt_classes, axis = 1) # (N,num_classes) -> (N,), where each element is the class index (highest score from its row)
  positive_indices = tf.squeeze(tf.where(class_indices > 0), axis = 1)  # (P,), tensor of P indices (the positive, non-background classes in class_indices)
  negative_indices = tf.squeeze(tf.where(class_indices <= 0), axis = 1) # (N,), tensor of N indices (the negative, background classes in class_indices)
  num_positive_proposals = tf.size(positive_indices)
  num_negative_proposals = tf.size(negative_indices)

  """
  def do_log1(x):
    tf.print("num_positive_proposals=", x, output_stream = "file:///projects/FasterRCNN/tf2/out.txt", summarize = -1)
    return x
  num_positive_proposals = Lambda(do_log1)(num_positive_proposals)
  def do_log2(x):
    tf.print("num_negative_proposals=", x, output_stream = "file:///projects/FasterRCNN/tf2/out.txt", summarize = -1)
    return x
  num_negative_proposals = Lambda(do_log2)(num_negative_proposals)
  """
  
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
  num_samples = tf.minimum(max_proposals, tf.size(class_indices))

  num_positive_samples = tf.minimum(tf.cast(tf.math.round(tf.cast(max_proposals, dtype = float) * positive_fraction), dtype = tf.int32), num_positive_proposals)
#  num_positive_samples = tf.minimum(tf.cast(tf.math.round(tf.cast(num_samples, dtype = float) * positive_fraction), dtype = num_samples.dtype), num_positive_proposals)
  num_negative_samples = tf.minimum(num_samples - num_positive_samples, num_negative_proposals)

  # Do we have enough?
  no_samples = tf.logical_or(tf.math.less_equal(num_positive_samples, 0), tf.math.less_equal(num_negative_samples, 0))

  # Sample randomly
  positive_sample_indices = tf.random.shuffle(positive_indices)[:num_positive_samples]
  negative_sample_indices = tf.random.shuffle(negative_indices)[:num_negative_samples]
  indices = tf.concat([ positive_sample_indices, negative_sample_indices ], axis = 0)
    
  """
  def do_log(x):
    tf.print("indices=", x, output_stream = "file:///projects/FasterRCNN/tf2/out.txt", summarize = -1)
    return x
  indices = Lambda(do_log)(indices)
  """

  # Return (if we have any samples)
  """
  # Because TF2/Keras is idiotic, these functions don't work because of an incompatibility between tf.function and KerasTensor 
  proposals = tf.cond(
    no_samples,
    true_fn = lambda: tf.zeros(shape = (0, 4), dtype = proposals.dtype),  # empty proposals tensor if no samples
    false_fn = lambda: tf.gather(proposals, indices = indices)            # gather samples
  )
  gt_classes = tf.cond(
    no_samples,
    true_fn = lambda: tf.zeros(shape = (0, tf.shape(gt_classes)[1]), dtype = gt_classes.dtype), # empty list of classes if no samples
    false_fn = lambda: tf.gather(gt_classes, indices = indices)                                 # gather samples
  )
  gt_regressions = tf.cond(
    no_samples,
    true_fn = lambda: tf.zeros(shape = (0, tf.shape(gt_regressions)[1]), dtype = gt_regressions.dtype), # empty list of classes if no samples
    false_fn = lambda: tf.gather(gt_regressions, indices = indices)                                     # gather samples
  )
  """
  return tf.gather(proposals, indices = indices), tf.gather(gt_classes, indices = indices), tf.gather(gt_regressions, indices = indices)
