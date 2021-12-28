import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import Model
from tensorflow.keras import Input

from . import vgg16
from . import rpn


def faster_rcnn_model(mode):
  assert mode == "train" or mode == "infer"
  if mode == "train":
    return _training_model()
  else:
    return _inference_model()

def _inference_model():
  #TODO: write me
  return None

def _training_model():
  image_shape_map = Input(shape = (3,), name = "image_shape_map")                         # holds shape of image: height, width, channels
  num_anchors = 9
  gt_rpn_map = Input(shape = (None, None, num_anchors, 6), name = "gt_rpn_map")           # (height, width, k, 6)
  anchor_map = Input(shape = (None, None, num_anchors * 4), name = "anchor_map")          # (height, width, k*4)
  anchor_valid_map = Input(shape = (None, None, num_anchors), name = "anchor_valid_map")  # (height, width, k)

  # Define model forward path
  stage1_feature_extractor_model = vgg16.conv_layers()
  rpn_score_output, rpn_regression_output, proposals_output = rpn.layers(
    input_map = stage1_feature_extractor_model.outputs[0],
    image_shape_map = image_shape_map,
    anchor_map = anchor_map,
    anchor_valid_map = anchor_valid_map,
    max_proposals_pre_nms = 12000,
    max_proposals_post_nms = 2000 
  )

  # Losses
  rpn_class_loss = rpn.class_loss(y_predicted = rpn_score_output, gt_rpn_map = gt_rpn_map)
  rpn_regression_loss = rpn.regression_loss(y_predicted = rpn_regression_output, gt_rpn_map = gt_rpn_map)

  # Build model
  model = Model([ stage1_feature_extractor_model.input, image_shape_map, anchor_map, anchor_valid_map, gt_rpn_map ], [ rpn_score_output, rpn_regression_output, proposals_output, rpn_class_loss, rpn_regression_loss ])
  model.add_loss(rpn_class_loss)
  model.add_loss(rpn_regression_loss)

  model.add_metric(rpn_class_loss, name = "rpn_class_loss")
  model.add_metric(rpn_regression_loss, name = "rpn_regression_loss")

  return model
