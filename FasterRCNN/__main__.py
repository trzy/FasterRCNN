# TODO:
# - How can regression loss by 0 with a randomly initialized network? Need to check these
#   examples by hand.
# - Test whether K.abs()/tf.abs() fail on Linux
# - Test loss function on an artificial y_true and y_predicted that we can compute by hand

#
# Faster R-CNN in Keras: https://towardsdatascience.com/faster-r-cnn-object-detection-implemented-by-keras-for-custom-data-from-googles-open-images-125f62b9141a
# Understanding RoI pooling: https://towardsdatascience.com/understanding-region-of-interest-part-1-roi-pooling-e4f5dd65bb44
# NMS for object detection: https://towardsdatascience.com/non-maximum-suppression-nms-93ce178e177c
#

#
# TODO:
# - Observed: the maximum number of positive anchors in in a VOC image is 102.
#   Is this a bug in our code? Paper talks about a 1:1 (128:128) ratio.
#

from . import utils
from . import visualization
from .dataset import VOC
from .models import vgg16
from .models import region_proposal_network

import argparse
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import Model
from tensorflow.keras import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import time

def rpn_loss_class_term_np(y_true, y_predicted):
  """
  NumPy reference implementation of loss function for immediate execution.
  """
  y_predicted_class = y_predicted
  y_true_class = y_true[:,:,:,:,1].reshape(y_predicted.shape)
  y_mask = y_true[:,:,:,:,0].reshape(y_predicted.shape)
  N_cls = float(np.count_nonzero(y_mask)) + 1e-9
  loss_all_anchors = -(y_true_class * np.log(y_predicted) + (1.0 - y_true_class) * np.log(1.0 - y_predicted)) # binary cross-entropy, element-wise
  relevant_loss_terms = y_mask * loss_all_anchors
  return np.sum(relevant_loss_terms) / N_cls

def rpn_loss_regression_term_np(y_true, y_predicted):
  y_predicted_regression = y_predicted
  y_true_regression = y_true[:,:,:,:,4:8].reshape(y_predicted.shape)
  mask_shape = (y_true.shape[0], y_true.shape[1], y_true.shape[2], y_true.shape[3])
  y_included = y_true[:,:,:,:,0].reshape(mask_shape)
  y_positive = y_true[:,:,:,:,1].reshape(mask_shape)
  y_mask = y_included * y_positive
  y_mask = np.repeat(y_mask, repeats = 4, axis = 3)
  N_cls = float(np.count_nonzero(y_included)) + 1e-9
  x = y_true_regression - y_predicted_regression
  x_abs = np.sqrt(x * x)  # K.abs/tf.abs crash (Windows only?)
  is_negative_branch = np.less(x_abs, 1.0).astype(np.float)
  R_negative_branch = 0.5 * x * x
  R_positive_branch = x_abs - 0.5
  loss_all_anchors = is_negative_branch * R_negative_branch + (1.0 - is_negative_branch) * R_positive_branch
  relevant_loss_terms = y_mask * loss_all_anchors
  return np.sum(relevant_loss_terms) / N_cls

def rpn_loss_class_term(y_true, y_predicted):
  y_predicted_class = tf.convert_to_tensor(y_predicted)
  y_true_class = tf.cast(tf.reshape(y_true[:,:,:,:,1], shape = tf.shape(y_predicted)), dtype = y_predicted.dtype)

  # y_true[:,:,:,0] is 1.0 for anchors included in the mini-batch
  y_mask = tf.cast(tf.reshape(y_true[:,:,:,:,0], shape = tf.shape(y_predicted)), dtype = y_predicted.dtype)

  # Compute how many anchors are actually used in the mini-batch (e.g.,
  # typically 256)
  N_cls = tf.cast(tf.math.count_nonzero(y_mask), dtype = tf.float32) + K.epsilon()

  # Compute element-wise loss for all anchors
  loss_all_anchors = K.binary_crossentropy(y_true_class, y_predicted_class)

  # Zero out the ones which should not have been included
  relevant_loss_terms = y_mask * loss_all_anchors

  # Sum the total loss and normalize by the number of anchors used
  return K.sum(relevant_loss_terms) / N_cls

def rpn_loss_regression_term(y_true, y_predicted):
  y_predicted_regression = tf.convert_to_tensor(y_predicted)
  y_true_regression = tf.cast(tf.reshape(y_true[:,:,:,:,4:8], shape = tf.shape(y_predicted)), dtype = y_predicted.dtype)
  
  # Include only anchors that are used in the mini-batch and which correspond
  # to objects (positive samples)
  mask_shape = tf.slice(tf.shape(y_true), begin = [0], size = [4])
  #mask_shape = (y_true.shape[0], y_true.shape[1], y_true.shape[2], y_true.shape[3])
  y_included = tf.cast(tf.reshape(y_true[:,:,:,:,0], mask_shape), dtype = y_predicted.dtype)
  y_positive = tf.cast(tf.reshape(y_true[:,:,:,:,1], mask_shape), dtype = y_predicted.dtype)
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
  x_abs = tf.sqrt(x * x)  # K.abs/tf.abs crash (Windows only?)
  is_negative_branch = tf.cast(tf.less(x_abs, 1.0), dtype = tf.float32)
  R_negative_branch = 0.5 * x * x
  R_positive_branch = x_abs - 0.5
  loss_all_anchors = is_negative_branch * R_negative_branch + (1.0 - is_negative_branch) * R_positive_branch

  # Zero out the ones which should not have been included
  relevant_loss_terms = y_mask * loss_all_anchors
  return K.sum(relevant_loss_terms) / N_cls

def build_rpn_model(input_image_shape = (None, None, 3)):
  conv_model = vgg16.conv_layers(input_shape = input_image_shape)
  classifier_output, regression_output = region_proposal_network.layers(input_map = conv_model.outputs[0])
  model = Model([conv_model.input], [classifier_output, regression_output])
  
  optimizer = Adam(lr=1e-5)
  loss = [ rpn_loss_class_term, rpn_loss_regression_term ]
  
  model.compile(optimizer = optimizer, loss = loss)
  return model

def train(voc):
  pass

def test_loss_functions(voc):
  model = build_rpn_model()
  train_data = voc.train_data(shuffle = False)

  print("Running loss function test over training samples...")

  # For each training sample, run forward inference pass and then compute loss
  # using both Keras backend and reference implementations
  max_diff_cls = 0
  max_diff_regr = 0
  epsilon = 1e-9
  for i in range(voc.num_samples["train"]):
    image_path, x, y = next(train_data)
    y = y.reshape((1, y.shape[0], y.shape[1], y.shape[2], y.shape[3]))  # convert to batch size of 1      
    x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))
    y_predicted_cls, y_predicted_regr = model.predict(x)
    loss_cls_keras  = K.eval(rpn_loss_class_term(y_true = K.variable(y), y_predicted = K.variable(y_predicted_cls)))
    loss_cls_np     = rpn_loss_class_term_np(y_true = y, y_predicted = y_predicted_cls)
    loss_regr_keras = K.eval(rpn_loss_regression_term(y_true = K.variable(y), y_predicted = K.variable(y_predicted_regr)))
    loss_regr_np    = rpn_loss_regression_term_np(y_true = y, y_predicted = y_predicted_regr)
    pct_diff_cls    = 100 * ((loss_cls_keras + epsilon) / (loss_cls_np + epsilon) - 1)    # epsilon because loss can be 0
    pct_diff_regr   = 100 * ((loss_regr_keras + epsilon) / (loss_regr_np + epsilon) - 1)
    print("loss_cls = %f %f\tloss_regr = %f %f\t%s" % (K.eval(loss_cls_keras), loss_cls_np, K.eval(loss_regr_keras), loss_regr_np, image_path))
    max_diff_cls = max(max_diff_cls, pct_diff_cls)
    max_diff_regr = max(max_diff_regr, pct_diff_regr)

  #print("Test succeeded -- Keras backend implementation is working")
  print("Max %% difference cls loss = %f" % max_diff_cls)
  print("Max %% difference regr loss = %f" % max_diff_regr)

# good test images:
# 2010_004041.jpg
# 2010_005080.jpg
if __name__ == "__main__":
  parser = argparse.ArgumentParser("FasterRCNN")
  parser.add_argument("--dataset-dir", metavar = "path", type = str, action = "store", default = "\\projects\\voc\\vocdevkit\\voc2012", help = "Dataset directory")
  parser.add_argument("--show-image", metavar = "file", type = str, action = "store", help = "Show an image with ground truth and corresponding anchor boxes")
  parser.add_argument("--train", action = "store_true", help = "Train the region proposal network")
  parser.add_argument("--test-loss", action = "store_true", help = "Test Keras backend implementation of loss functions")
  options = parser.parse_args()

  voc = VOC(dataset_dir = options.dataset_dir, scale = 600)
  
  if options.show_image:
    info = voc.get_image_description(path = voc.get_full_path(options.show_image))

    # Need to build the model for this image size in order to be able to visualize boxes correctly
    conv_model = vgg16.conv_layers(input_shape = (info.height,info.width,3))
    classifier_output, regression_output = region_proposal_network.layers(input_map = conv_model.outputs[0])
    model = Model([conv_model.input], [classifier_output, regression_output])

    print(classifier_output.shape, model.input.shape)
    
    visualization.show_annotated_image(voc = voc, filename = options.show_image, draw_anchor_intersections = True, image_input_map = model.input, anchor_map = classifier_output)

  if options.test_loss:
    test_loss_functions(voc)
    
  if options.train:
    
    #model = build_rpn_model()
    #info = voc.get_image_description(voc.get_full_path("2008_000019.jpg"))
    #x = info.load_image_data()
    #x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))
    #model.predict(x)
    #print("ok")
    #exit()
    
    model = build_rpn_model()
    train_data = voc.train_data(limit_samples = 16)

    num_epochs = 16    
    for epoch in range(num_epochs):
      progbar = tf.keras.utils.Progbar(voc.num_samples["train"])
      print("Epoch %d/%d" % (epoch + 1, num_epochs))
    
      for i in range(voc.num_samples["train"]):
        image_path, x, y = next(train_data)
        y = y.reshape((1, y.shape[0], y.shape[1], y.shape[2], y.shape[3]))  # convert to batch size of 1      
        x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))
        loss = model.train_on_batch(x = x, y = y) # loss = [sum, loss_cls, loss_regr]
        progbar.update(current = i, values = [ ("loss", loss[0]) ])

    

