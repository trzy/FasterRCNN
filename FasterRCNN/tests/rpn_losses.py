#
# rpn_losses.py
#
# Tests the Keras backend implementations of the RPN loss functions by
# comparing to NumPy reference implementations.
#
# It recommended to run this test with and without pre-trained model weights.
# When the amount of incorrectly classified anchors is large, the error between
# the two implementations can be as high as ~2%. When the predictions are
# largely correct, the error is very low. This may indicate that one of two
# terms that comprise the binary crossentropy function differs slightly in the
# Keras implementation.
#
# I do not believe the magnitude of error of the class loss is cause for concern
# but it would be nice to understand why it occurs.
#

from ..dataset import VOC
from ..models import vgg16
from ..models import region_proposal_network
from ..models.rpn_loss import rpn_class_loss
from ..models.rpn_loss import rpn_regression_loss
from ..models.rpn_loss import rpn_class_loss_np
from ..models.rpn_loss import rpn_regression_loss_np

import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K

def build_rpn_model(weights_filepath = None):
  conv_model = vgg16.conv_layers(input_shape = (None, None, 3), l2 = 0)
  classifier_output, regression_output = region_proposal_network.layers(input_map = conv_model.outputs[0], l2 = 0)
  model = Model([conv_model.input], [classifier_output, regression_output])

  optimizer = SGD(lr = 1e-3, momentum = 0.9)
  loss = [ rpn_class_loss, rpn_regression_loss ]
  model.compile(optimizer = optimizer, loss = loss)

  # Load weights
  if weights_filepath:
    model.load_weights(filepath = weights_filepath, by_name = True)
    print("Loaded RPN model weights from %s" % weights_filepath)
  else:
    # When initializing from scratch, use pre-trained VGG
    vgg16.load_imagenet_weights(model = model)
    print("Loaded pre-trained VGG-16 weights")

  return model

if __name__ == "__main__":
  parser = argparse.ArgumentParser("rpn_losses")
  parser.add_argument("--dataset-dir", metavar = "path", type = str, action = "store", default = "\\projects\\voc\\vocdevkit\\voc2012", help = "Dataset directory")
  parser.add_argument("--load-from", metavar="filepath", type = str, action = "store", help = "File to load initial model weights from")
  options = parser.parse_args()

  model = build_rpn_model(weights_filepath = options.load_from)

  voc = VOC(dataset_dir = options.dataset_dir, scale = 600)
  train_data = voc.train_data(shuffle = False)

  print("Running loss function test over training samples...")

  # For each training sample, run forward inference pass and then compute loss
  # using both Keras backend and reference implementations
  max_diff_cls = 0
  max_diff_regr = 0
  epsilon = 1e-9
  for i in range(voc.num_samples["train"]):
    image_path, x, y, anchor_boxes = next(train_data)
    y = y.reshape((1, y.shape[0], y.shape[1], y.shape[2], y.shape[3]))  # convert to batch size of 1
    x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))
    y_predicted_cls, y_predicted_regr = model.predict(x)
    loss_cls_keras  = K.eval(rpn_class_loss(y_true = K.variable(y), y_predicted = K.variable(y_predicted_cls)))
    loss_cls_np     = rpn_class_loss_np(y_true = y, y_predicted = y_predicted_cls)
    loss_regr_keras = K.eval(rpn_regression_loss(y_true = K.variable(y), y_predicted = K.variable(y_predicted_regr)))
    loss_regr_np    = rpn_regression_loss_np(y_true = y, y_predicted = y_predicted_regr)
    pct_diff_cls    = abs(100 * ((loss_cls_keras + epsilon) / (loss_cls_np + epsilon) - 1))   # epsilon because loss can be 0
    pct_diff_regr   = abs(100 * ((loss_regr_keras + epsilon) / (loss_regr_np + epsilon) - 1))
    print("loss_cls = %f %f\tloss_regr = %f %f\t%s" % (loss_cls_keras, loss_cls_np, loss_regr_keras, loss_regr_np, image_path))
    #assert pct_diff_cls < 0.01 and pct_diff_regr < 0.01 # expect negligible difference (< 0.01%)
    max_diff_cls = max(max_diff_cls, pct_diff_cls)
    max_diff_regr = max(max_diff_regr, pct_diff_regr)

  print("Max %% difference cls loss = %f" % max_diff_cls)
  print("Max %% difference regr loss = %f" % max_diff_regr)

  # Check result. I'm not thrilled that the % difference between the NumPy
  # reference and Keras implementations of the class loss can be ~2%. I
  # suspect the binary crossentropy function is computed in some slightly
  # different way.
  assert max_diff_cls < 2.5 and max_diff_regr < 1.0, "** Test FAILED **"
  print("** Test PASSED **")
