# TODO next:
# - See TODO in compute_ground_truth_regressions -- are we ever stealing any anchors away?
#   If so, we need to be more careful about which regression parameters we choose. We may
#   need a second pass that, for each positive anchor assigned to multiple boxes, chooses
#   the highest IoU box to regress against.
# - Check to see if NMS code is correct and maybe implement an alternative.
# - Re-add regression. Visualize proposals by showing GT box regions as green

# - Why are class and regression losses so different? Should be comparable.
# - Test whether K.abs()/tf.abs() fail on Linux
# - Weight decay, dropout, momentum

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
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
import time

#TODO: turn this into a unit test
def test_loss_manual():
  # Creates an artificial prediction and ground truth pair to test the box
  # regression loss function. All anchors are marked as valid and a single
  # anchor is marked as positive. Only it should be reflected in the loss.

  predicted_t = np.array([10, 20, 30, 40])
  true_t      = np.array([11, 22, 33, 44])

  y_predicted = np.zeros((1, 2, 2, 9*4))
  y_true      = np.zeros((1, 2, 2, 9, 8))

  y_predicted[0,0,0,0:9*4] = -50 * np.ones(9*4)   # all anchors at (0,0) are invalid
  y_predicted[0,1,0,0:9*4] = -60 * np.ones(9*4)   # ""
  y_predicted[0,1,1,0:9*4] = -100 * np.ones(9*4)  # ""
  y_predicted[0,0,1,0:9*4] = -1 * np.ones(9*4)    # anchors #0, #2-#8 at position (0,1) are invalid and all regression values set to -1
  y_predicted[0,0,1,4:8] = predicted_t            # anchor #1 is valid

  # Make all anchors valid (but not positive)
  y_true[0,:,:,:,0] = 1.0

  # Anchor #1 at (0,1)
  y_true[0,0,1,1,0] = 1.0         # is valid
  y_true[0,0,1,1,1] = 1.0         # is positive example
  y_true[0,0,1,1,2] = 1.0 + 2.0   # box number 2
  y_true[0,0,1,1,3] = 0.85        # IoU
  y_true[0,0,1,1,4] = true_t[0]   # ty
  y_true[0,0,1,1,5] = true_t[1]   # tx
  y_true[0,0,1,1,6] = true_t[2]   # th
  y_true[0,0,1,1,7] = true_t[3]   # tw

  loss = rpn_loss_regression_term_np(y_true, y_predicted)

  print("Computed Loss =", loss)

  # Compute by hand for the one valid (positive && valid) anchor that we have
  x = true_t - predicted_t
  x_abs = np.abs(x)
  is_negative_branch = np.less(x_abs, 1.0).astype(np.float)
  R_negative_branch = 0.5 * x * x
  R_positive_branch = x_abs - 0.5
  loss_all_components = is_negative_branch * R_negative_branch + (1.0 - is_negative_branch) * R_positive_branch
  N_cls = 2 * 2 * 9 # all anchors included in computation
  loss_expected = np.sum(loss_all_components) / N_cls

  print("Expected Loss = ", loss_expected)


def rpn_loss_class_term_np(y_true, y_predicted):
  """
  NumPy reference implementation of loss function for immediate execution.
  """
  y_predicted_class = y_predicted
  y_true_class = y_true[:,:,:,:,1].reshape(y_predicted.shape)
  y_mask = y_true[:,:,:,:,0].reshape(y_predicted.shape)
  epsilon = 1e-9
  N_cls = float(np.count_nonzero(y_mask)) + epsilon
  loss_all_anchors = -(y_true_class * np.log(y_predicted + epsilon) + (1.0 - y_true_class) * np.log(1.0 - y_predicted + epsilon)) # binary cross-entropy, element-wise
  relevant_loss_terms = y_mask * loss_all_anchors
  print(np.sum(loss_all_anchors), np.sum(relevant_loss_terms))
  return np.sum(relevant_loss_terms) / N_cls

def rpn_loss_regression_term_np(y_true, y_predicted):
  scale_factor = 1.0
  sigma = 3.0
  sigma_squared = sigma * sigma
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
  R_negative_branch = 0.5 * sigma_squared * x * x
  R_positive_branch = x_abs - 0.5 / sigma_squared
  loss_all_anchors = is_negative_branch * R_negative_branch + (1.0 - is_negative_branch) * R_positive_branch
  relevant_loss_terms = y_mask * loss_all_anchors
  return scale_factor * np.sum(relevant_loss_terms) / N_cls

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
  #TODO: factor this out as an actual conifgurable parameter and make this function return a loss function
  scale_factor = 1.0  # hyper-parameter that controls magnitude of regression loss and is chosen to make regression term comparable to class term
  sigma = 3.0         # see: https://github.com/rbgirshick/py-faster-rcnn/issues/89
  sigma_squared = sigma * sigma

  y_predicted_regression = tf.convert_to_tensor(y_predicted)
  y_true_regression = tf.cast(tf.reshape(y_true[:,:,:,:,4:8], shape = tf.shape(y_predicted)), dtype = y_predicted.dtype)

  # Include only anchors that are used in the mini-batch and which correspond
  # to objects (positive samples)
  mask_shape = tf.slice(tf.shape(y_true), begin = [0], size = [4])
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
  R_negative_branch = 0.5 * x * x * sigma_squared
  R_positive_branch = x_abs - 0.5 / sigma_squared
  loss_all_anchors = is_negative_branch * R_negative_branch + (1.0 - is_negative_branch) * R_positive_branch

  # Zero out the ones which should not have been included
  relevant_loss_terms = y_mask * loss_all_anchors
  return scale_factor * K.sum(relevant_loss_terms) / N_cls

def print_weights(model):
  for layer in model.layers:
    weights = layer.get_weights()
    if len(weights) > 0:
      print(layer.name, layer.get_weights()[0][0])

def build_rpn_model(learning_rate, input_image_shape = (None, None, 3), weights_filepath = None, l2 = 0):
  conv_model = vgg16.conv_layers(input_shape = input_image_shape, l2 = l2)
  classifier_output, regression_output = region_proposal_network.layers(input_map = conv_model.outputs[0], l2 = l2)
  model = Model([conv_model.input], [classifier_output, regression_output])
  #model = Model([conv_model.input], [classifier_output ])

  optimizer = SGD(lr=learning_rate, momentum=0.9)
  loss = [ rpn_loss_class_term, rpn_loss_regression_term ]
  #loss = [ rpn_loss_class_term ]
  model.compile(optimizer = optimizer, loss = loss)

  # Load before freezing layers
  if weights_filepath:
    model.load_weights(filepath = weights_filepath, by_name = True)
    print("Loaded model weights from %s" % weights_filepath)
  else:
    # When initializing from scratch, use pre-trained VGG
    vgg16.load_imagenet_weights(model = model)

  # Freeze first two convolutional blocks during training
  utils.freeze_layers(model = model, layers = "block1_conv1, block1_conv2, block2_conv1, block2_conv2")
  return model

def train(voc):
  pass

def show_image(voc, filename):
  info = voc.get_image_description(path = voc.get_full_path(filename))

  # Need to build the model for this image size in order to be able to visualize boxes correctly
  conv_model = vgg16.conv_layers(input_shape = (info.height,info.width,3))
  classifier_output, regression_output = region_proposal_network.layers(input_map = conv_model.outputs[0])
  model = Model([conv_model.input], [classifier_output, regression_output])

  visualization.show_annotated_image(voc = voc, filename = options.show_image, draw_anchor_intersections = True, image_input_map = model.input, anchor_map = classifier_output)

def infer_boxes(model, voc, filename):
  info = voc.get_image_description(path = voc.get_full_path(filename))
  x = info.load_image_data()
  x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))
  #y_class = model.predict(x)
  #y_regression = np.zeros((y_class.shape[0], y_class.shape[1], y_class.shape[2], y_class.shape[3] * 4))
  y_class, y_regression = model.predict(x)
  for yy in range(y_class.shape[1]):
    for xx in range(y_class.shape[2]):
      for kk in range(y_class.shape[3]):
        if y_class[0,yy,xx,kk] > 0.5:
          print("%d,%d,%d -> %f" % (yy, xx, kk, y_class[0,yy,xx,kk]))
  y_true = info.get_complete_ground_truth_regressions_map()
  y_true = y_true.reshape((1, y_true.shape[0], y_true.shape[1], y_true.shape[2], y_true.shape[3]))
  print("loss=", rpn_loss_class_term_np(y_true=y_true, y_predicted=y_class))
  visualization.show_proposed_regions(voc = voc, filename = filename, y_true = y_true, y_class = y_class, y_regression = y_regression)

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
    assert pct_diff_cls < 0.01 and pct_diff_regr < 0.01 # expect negligible difference (< 0.01%)
    max_diff_cls = max(max_diff_cls, pct_diff_cls)
    max_diff_regr = max(max_diff_regr, pct_diff_regr)

  print("Test succeeded -- Keras backend implementation is working")
  print("Max %% difference cls loss = %f" % max_diff_cls)
  print("Max %% difference regr loss = %f" % max_diff_regr)

# good test images:
# 2010_004041.jpg
# 2010_005080.jpg
# 2008_000019.jpg
# 2009_004872.jpg
if __name__ == "__main__":
  parser = argparse.ArgumentParser("FasterRCNN")
  parser.add_argument("--dataset-dir", metavar = "path", type = str, action = "store", default = "\\projects\\voc\\vocdevkit\\voc2012", help = "Dataset directory")
  parser.add_argument("--show-image", metavar = "file", type = str, action = "store", help = "Show an image with ground truth and corresponding anchor boxes")
  parser.add_argument("--train", action = "store_true", help = "Train the region proposal network")
  parser.add_argument("--epochs", metavar = "count", type = utils.positive_int, action = "store", default = "10", help = "Number of epochs to train for")
  parser.add_argument("--learning-rate", metavar = "rate", type = float, action = "store", default = "0.001", help = "Learning rate")
  parser.add_argument("--mini-batch", metavar = "size", type = utils.positive_int, action = "store", default = "256", help = "Mini-batch size")
  parser.add_argument("--l2", metavar = "value", type = float, action = "store", default = "2.5e-4", help = "L2 regularization")
  parser.add_argument("--save-to", metavar = "filepath", type = str, action = "store", help = "File to save model weights to when training is complete")
  parser.add_argument("--load-from", metavar="filepath", type = str, action = "store", help = "File to load initial model weights from")
  parser.add_argument("--test-loss", action = "store_true", help = "Test Keras backend implementation of loss functions")
  parser.add_argument("--infer-boxes", metavar = "file", type = str, action = "store", help = "Run inference on image using region proposal network and display bounding boxes")
  options = parser.parse_args()

  voc = VOC(dataset_dir = options.dataset_dir, scale = 600)

  model = build_rpn_model(weights_filepath = options.load_from, learning_rate = options.learning_rate, l2 = options.l2)
  model.summary()

  if options.show_image:
    show_image(voc = voc, filename = options.show_image)

  if options.test_loss:
    test_loss_functions(voc)

  if options.infer_boxes:
    infer_boxes(model = model, voc = voc, filename = options.infer_boxes)

  if options.train:

    #model = build_rpn_model()
    #info = voc.get_image_description(voc.get_full_path("2008_000019.jpg"))
    #x = info.load_image_data()
    #x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))
    #model.predict(x)
    #print("ok")
    #exit()

    train_data = voc.train_data(cache_images = True, mini_batch_size = options.mini_batch)
    num_samples = voc.num_samples["train"]  # number of iterations in an epoch

    rpn_total_losses = np.zeros(num_samples)
    class_losses = np.zeros(num_samples)
    regression_losses = np.zeros(num_samples)
    class_accuracies = np.zeros(num_samples)

    for epoch in range(options.epochs):
      progbar = tf.keras.utils.Progbar(num_samples)
      print("Epoch %d/%d" % (epoch + 1, options.epochs))

      for i in range(num_samples):
        # Fetch one sample and reshape to batch size of 1
        image_path, x, y = next(train_data)
        y = y.reshape((1, y.shape[0], y.shape[1], y.shape[2], y.shape[3]))  # convert to batch size of 1
        x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))

        # Back prop one step
        losses = model.train_on_batch(x = x, y = y) # loss = [sum, loss_cls, loss_regr]
        #loss = model.train_on_batch(x = x, y = y) # loss = [sum, loss_cls, loss_regr]



        ## Predict to compute current accuracy
        #y_predicted_class = model.predict_on_batch(x = x)
        #y_true_class = y[:,:,:,:,2].reshape(y_predicted_class.shape)
        #y_valid = y[:,:,:,:,0].reshape(y_predicted_class.shape)
        #assert np.size(y_true_class) == np.size(y_predicted_class)
        #ground_truth_positives = np.where(y_true_class > 0, True, False)
        #ground_truth_negatives = np.where(y_true_class < 0, True, False)
        #true_positives = np.sum(np.where(y_predicted_class > 0.5, True, False) * ground_truth_positives)
        #true_negatives = np.sum(np.where(y_predicted_class < 0.5, True, False) * ground_truth_negatives)
        #total_samples = np.sum(ground_truth_positives) + np.sum(ground_truth_negatives)
        #class_accuracy = (true_positives + true_negatives) / total_samples

        ## Save losses for this iteration and update mean
        #class_losses[i] = loss
        #class_accuracies[i] = class_accuracy
        #mean_class_loss = np.mean(class_losses[0:i+1])
        #mean_class_accuracy = np.mean(class_accuracies[0:i+1])

        ## Progress
        #progbar.update(current = i, values = [ ("class_loss", mean_class_loss), ("class_accuracy", mean_class_accuracy) ])

        #mean_rpn_total_loss = mean_class_loss


        # Predict to compute current accuracy
        #y_predicted_class, y_predicted_regression = model.predict_on_batch(x = x)
        #y_true_class = y[:,:,:,:,1].reshape(y_predicted_class.shape)
        #y_valid = y[:,:,:,:,0].reshape(y_predicted_class.shape)
        #assert np.size(y_true_class) == np.size(y_predicted_class)
        #true_positives = np.sum(y_valid * np.where(y_predicted_class > 0.5, True, False) * np.where(y_true_class > 0.5, True, False))
        #true_negatives = np.sum(y_valid * np.where(y_predicted_class < 0.5, True, False) * np.where(y_true_class < 0.5, True, False))
        #class_accuracy = (true_positives + true_negatives) / np.size(y_predicted_class)
        y_predicted_class, y_predicted_regression = model.predict_on_batch(x = x)
        y_true_class = y[:,:,:,:,2].reshape(y_predicted_class.shape)
        y_valid = y[:,:,:,:,0].reshape(y_predicted_class.shape)
        assert np.size(y_true_class) == np.size(y_predicted_class)
        ground_truth_positives = np.where(y_true_class > 0, True, False)
        ground_truth_negatives = np.where(y_true_class < 0, True, False)
        true_positives = np.sum(np.where(y_predicted_class > 0.5, True, False) * ground_truth_positives)
        true_negatives = np.sum(np.where(y_predicted_class < 0.5, True, False) * ground_truth_negatives)
        total_samples = np.sum(ground_truth_positives) + np.sum(ground_truth_negatives)
        class_accuracy = (true_positives + true_negatives) / total_samples

        # Save losses for this iteration and update mean
        rpn_total_losses[i] = losses[0]
        class_losses[i] = losses[1]
        regression_losses[i] = losses[2]
        class_accuracies[i] = class_accuracy
        mean_class_loss = np.mean(class_losses[0:i+1])
        mean_regression_loss = np.mean(regression_losses[0:i+1])
        mean_rpn_total_loss = mean_class_loss + mean_regression_loss
        mean_class_accuracy = np.mean(class_accuracies[0:i+1])

        # Progress
        progbar.update(current = i, values = [ ("rpn_total_loss", mean_rpn_total_loss), ("class_loss", mean_class_loss), ("regression_loss", mean_regression_loss), ("class_accuracy", mean_class_accuracy) ])

      # Checkpoint
      print("")
      checkpoint_filename = "checkpoint-%d-%1.2f.hdf5" % (epoch, mean_rpn_total_loss)
      model.save_weights(filepath = checkpoint_filename, overwrite = True, save_format = "h5")
      print("Saved checkpoint: %s" % checkpoint_filename)

    # Save learned model parameters
    if options.save_to is not None:
      model.save_weights(filepath = options.save_to, overwrite = True, save_format = "h5")
      print("Saved model weights to %s" % options.save_to)
