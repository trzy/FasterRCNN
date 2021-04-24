# TODO:
#
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
from .models.rpn_loss import rpn_class_loss
from .models.rpn_loss import rpn_regression_loss
from .models import classifier_network

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

def print_weights(model):
  for layer in model.layers:
    weights = layer.get_weights()
    if len(weights) > 0:
      print(layer.name, layer.get_weights()[0][0])

def build_rpn_model(learning_rate, clipnorm, input_image_shape = (None, None, 3), weights_filepath = None, l2 = 0, freeze = False):
  conv_model = vgg16.conv_layers(input_shape = input_image_shape, l2 = l2)
  classifier_output, regression_output = region_proposal_network.layers(input_map = conv_model.outputs[0], l2 = l2)
  model = Model([conv_model.input], [classifier_output, regression_output])

  optimizer = SGD(lr = learning_rate, momentum = 0.9, clipnorm = clipnorm)
  loss = [ rpn_class_loss, rpn_regression_loss ]
  model.compile(optimizer = optimizer, loss = loss)

  # Load before freezing layers
  if weights_filepath:
    model.load_weights(filepath = weights_filepath, by_name = True)
    print("Loaded RPN model weights from %s" % weights_filepath)
  else:
    # When initializing from scratch, use pre-trained VGG
    vgg16.load_imagenet_weights(model = model)
    print("Loaded pre-trained VGG-16 weights")

  # Freeze first two convolutional blocks during training
  if freeze:
    utils.freeze_layers(model = model, layers = "block1_conv1, block1_conv2, block2_conv1, block2_conv2")
  return model, conv_model

def build_classifier_model(conv_model, learning_rate, clipnorm, weights_filepath = None):
  proposal_boxes = Input(shape = (None, 4), dtype = tf.int32)
  classifier_model = classifier_network.layers(input_map = conv_model.outputs[0], proposal_boxes = proposal_boxes)
  model = Model([conv_model.input, proposal_boxes], [classifier_model])

  optimizer = SGD(lr = learning_rate, momentum = 0.9, clipnorm = clipnorm)
  model.compile(optimizer = optimizer, loss = "binary_crossentropy")  #TODO: just a placeholder loss for now

  # Load weights
  if weights_filepath:
    model.load_weights(filepath = weights_filepath, by_name = True)
    print("Loaded classifier model weights from %s" % weights_filepath)

  return model

def show_image(voc, filename):
  info = voc.get_image_description(path = voc.get_full_path(filename))

  # Need to build the model for this image size in order to be able to visualize boxes correctly
  conv_model = vgg16.conv_layers(input_shape = (info.height,info.width,3))
  classifier_output, regression_output = region_proposal_network.layers(input_map = conv_model.outputs[0])
  model = Model([conv_model.input], [classifier_output, regression_output])

  visualization.show_annotated_image(voc = voc, filename = options.show_image, draw_anchor_intersections = True, image_input_map = model.input, anchor_map = classifier_output)

def infer_boxes(model, voc, filename):
  from .models.rpn_loss import rpn_class_loss_np
  info = voc.get_image_description(path = voc.get_full_path(filename))
  x = info.load_image_data()
  x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))
  y_class, y_regression = model.predict(x)
  for yy in range(y_class.shape[1]):
    for xx in range(y_class.shape[2]):
      for kk in range(y_class.shape[3]):
        if y_class[0,yy,xx,kk] > 0.5:
          print("%d,%d,%d -> %f" % (yy, xx, kk, y_class[0,yy,xx,kk]))
  y_true = info.get_complete_ground_truth_regressions_map()
  y_true = y_true.reshape((1, y_true.shape[0], y_true.shape[1], y_true.shape[2], y_true.shape[3]))
  print("class loss=", rpn_class_loss_np(y_true=y_true, y_predicted=y_class))
  visualization.show_proposed_regions(voc = voc, filename = filename, y_true = y_true, y_class = y_class, y_regression = y_regression)

class TrainingStatistics:
  def __init__(self):
    self._step_number = 0

    self._rpn_total_losses = np.zeros(num_samples)
    self._rpn_class_losses = np.zeros(num_samples)
    self._rpn_regression_losses = np.zeros(num_samples)
    self._rpn_class_accuracies = np.zeros(num_samples)
    self._rpn_class_recalls = np.zeros(num_samples)

    self.rpn_mean_class_loss = float("inf")
    self.rpn_mean_class_accuracy = 0
    self.rpn_mean_class_recall = 0
    self.rpn_mean_regression_loss = float("inf")
    self.rpn_mean_total_loss = float("inf")

  def on_epoch_begin(self):
    """
    Must be called at the beginning of each epoch.
    """
    self._step_number = 0

  def on_epoch_end(self):
    """
    Must be called at the end of each epoch after the last step.
    """
    pass

  def on_step_begin(self):
    """
    Must be called at the beginning of each training step before the other step
    update functions (e.g., on_rpn_step()).
    """
    pass

  def on_rpn_step(self, losses, y_predicted_class, y_predicted_regression, y_true_minibatch, y_true):
    """
    Must be called on each training step after the RPN model has been updated.
    Updates the training statistics for the RPN model.

    Parameters:
 
      losses: RPN model losses from Keras train_on_batch() as a 3-element array,
        [ total_loss, class_loss, regression_loss ]
      y_predicted_class: RPN model objectness classification output of shape
        (1, height, width, k), where k is the number of anchors. Each element
        indicates the corresponding anchor is an object (>0.5) or background
        (<0.5).
      y_predicted_regression: RPN model regression outputs, with shape
        (1, height, width, k*4).
      y_true_minibatch: RPN ground truth map for the mini-batch used in this
        training step. The map contains ground truth regression targets and
        object classes and, most importantly, a mask indicating which anchors
        are valid and were used in the mini-batch. See
        region_proposal_network.compute_anchor_label_assignments() for layout.
      y_true: Complete RPN ground truth map for all anchors in the image (the
        anchor valid mask indicates all valid anchors from which mini-batches
        are drawn). This is used to compute classification accuracy and recall
        statistics because predictions occur over all possible anchors in the
        image.
    """
    y_true_class = y_true[:,:,:,:,2].reshape(y_predicted_class.shape)  # ground truth classes
    y_valid = y_true_minibatch[:,:,:,:,0].reshape(y_predicted_class.shape)      # valid anchors participating in this mini-batch
    assert np.size(y_true_class) == np.size(y_predicted_class)
    
    # Compute class accuracy and recall
    ground_truth_positives = np.where(y_true_class > 0, True, False)
    ground_truth_negatives = np.where(y_true_class < 0, True, False)
    num_ground_truth_positives = np.sum(ground_truth_positives)
    num_ground_truth_negatives = np.sum(ground_truth_negatives)
    true_positives = np.sum(np.where(y_predicted_class > 0.5, True, False) * ground_truth_positives)
    true_negatives = np.sum(np.where(y_predicted_class < 0.5, True, False) * ground_truth_negatives)
    total_samples = num_ground_truth_positives + num_ground_truth_negatives
    class_accuracy = (true_positives + true_negatives) / total_samples
    class_recall = true_positives / num_ground_truth_positives

    # Update progress
    i = self._step_number
    self._rpn_total_losses[i] = losses[0]
    self._rpn_class_losses[i] = losses[1]
    self._rpn_regression_losses[i] = losses[2]
    self._rpn_class_accuracies[i] = class_accuracy
    self._rpn_class_recalls[i] = class_recall
    
    self.rpn_mean_class_loss = np.mean(self._rpn_class_losses[0:i+1])
    self.rpn_mean_class_accuracy = np.mean(self._rpn_class_accuracies[0:i+1])
    self.rpn_mean_class_recall = np.mean(self._rpn_class_recalls[0:i+1])
    self.rpn_mean_regression_loss = np.mean(self._rpn_regression_losses[0:i+1])
    self.rpn_mean_total_loss = self.rpn_mean_class_loss + self.rpn_mean_regression_loss


  def on_step_end(self):
    """
    Must be called at the end of each training step after all the other step functions.
    """
    self._step_number += 1

def convert_proposals_to_classifier_network_format(proposals, input_image_shape, cnn_output_shape, ground_truth_object_boxes, num_classes):
  """
  Converts proposals from (N,5) shaped map, containing proposal box corners and
  objectness class score, to (1,M,4) format. Proposals are converted to anchor
  map space from input image pixel space, clipped (hence why M <= N), and
  converted to (y_min,x_min,height,width) format. Also returns a (1,M,C) shaped
  tensor of one-hot encoded class labels for each proposal, where C is the
  number of classes (including the background class, 0).
  """
  # Strip out class score
  proposals = proposals[:,0:4]

  # Perform clipping in RPN map space so that RoI pooling layer is never
  # passed rectangles that exceed the boundaries of its input map
  proposals = region_proposal_network.clip_box_coordinates_to_map_boundaries(boxes = proposals, map_shape = input_image_shape)

  # Generate one-hot labels for each proposal
  y_true_proposal_classes = region_proposal_network.label_proposals(proposals = proposals, ground_truth_object_boxes = ground_truth_object_boxes, num_classes = num_classes)
  
  # Convert to anchor map (RPN output map) space
  proposals = vgg16.convert_box_coordinates_from_image_to_output_map_space(box = proposals, output_map_shape = cnn_output_shape)

  # Convert from (y_min,x_min,y_max,x_max) -> (y_min,x_min,height,width) as expected by RoI pool layer
  proposals[:,2:4] = proposals[:,2:4] - proposals[:,0:2] + 1

  # Reshape to batch size of 1
  proposals = proposals.reshape((1, proposals.shape[0], proposals.shape[1]))
  y_true_proposal_classes = y_true_proposal_classes.reshape((1, y_true_proposal_classes.shape[0], y_true_proposal_classes.shape[1]))

  return proposals, y_true_proposal_classes

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
  parser.add_argument("--clipnorm", metavar = "value", type = float, action = "store", default = "1.0", help = "Clip gradient norm to value")
  parser.add_argument("--mini-batch", metavar = "size", type = utils.positive_int, action = "store", default = "256", help = "Mini-batch size")
  parser.add_argument("--l2", metavar = "value", type = float, action = "store", default = "2.5e-4", help = "L2 regularization")
  parser.add_argument("--freeze", action = "store_true", help = "Freeze first 2 blocks of VGG-16")
  parser.add_argument("--save-to", metavar = "filepath", type = str, action = "store", help = "File to save model weights to when training is complete")
  parser.add_argument("--load-from", metavar="filepath", type = str, action = "store", help = "File to load initial model weights from")
  parser.add_argument("--infer-boxes", metavar = "file", type = str, action = "store", help = "Run inference on image using region proposal network and display bounding boxes")
  options = parser.parse_args()

  voc = VOC(dataset_dir = options.dataset_dir, scale = 600)

  rpn_model, conv_model = build_rpn_model(weights_filepath = options.load_from, learning_rate = options.learning_rate, clipnorm = options.clipnorm, l2 = options.l2)
  classifier_model = build_classifier_model(conv_model = conv_model, learning_rate = options.learning_rate, clipnorm = options.clipnorm)
  rpn_model.summary()

  if options.show_image:
    show_image(voc = voc, filename = options.show_image)

  if options.infer_boxes:
    infer_boxes(model = rpn_model, voc = voc, filename = options.infer_boxes)

  if options.train:
    train_data = voc.train_data(cache_images = True, mini_batch_size = options.mini_batch)
    num_samples = voc.num_samples["train"]  # number of iterations in an epoch

    stats = TrainingStatistics()

    for epoch in range(options.epochs):
      stats.on_epoch_begin()
      progbar = tf.keras.utils.Progbar(num_samples)
      print("Epoch %d/%d" % (epoch + 1, options.epochs))

      for i in range(num_samples):
        stats.on_step_begin()

        # Fetch one sample and reshape to batch size of 1
        # TODO: should we just return complete y_true with a y_batch/y_valid map to define mini-batch?
        image_path, x, y_true_minibatch, anchor_boxes = next(train_data)
        input_image_shape = x.shape
        cnn_output_shape = vgg16.compute_output_map_shape(input_image_shape = input_image_shape)
        image_info = voc.get_image_description(image_path)
        ground_truth_object_boxes = image_info.get_boxes()              #TODO: return this from iterator so we don't need image_info
        y_true = image_info.get_complete_ground_truth_regressions_map() #TODO: ""
        y_true = y_true.reshape((1, y_true.shape[0], y_true.shape[1], y_true.shape[2], y_true.shape[3]))
        y_true_minibatch = y_true_minibatch.reshape((1, y_true_minibatch.shape[0], y_true_minibatch.shape[1], y_true_minibatch.shape[2], y_true_minibatch.shape[3]))  # convert to batch size of 1
        x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))

        # RPN: back prop one step (and then predict so we can evaluate accuracy)
        rpn_losses = rpn_model.train_on_batch(x = x, y = y_true_minibatch) # loss = [sum, loss_cls, loss_regr]
        y_predicted_class, y_predicted_regression = rpn_model.predict_on_batch(x = x)

        # Test: run classifier model forward (not yet complete)
        proposals = region_proposal_network.extract_proposals(y_predicted_class = y_predicted_class, y_predicted_regression = y_predicted_regression, y_true = y_true, anchor_boxes = anchor_boxes)
        if proposals.shape[0] > 0:
          # Prepare proposals for input to classifier network and generate
          # labels
          proposals, y_true_proposal_classes = convert_proposals_to_classifier_network_format(
            proposals = proposals,
            input_image_shape = input_image_shape,
            cnn_output_shape = cnn_output_shape,
            ground_truth_object_boxes = ground_truth_object_boxes,
            num_classes = voc.num_classes)

          # Classifier forward pass
          y_final = classifier_model.predict_on_batch(x = [ x, proposals ])

        # Update progress
        stats.on_rpn_step(losses = rpn_losses, y_predicted_class = y_predicted_class, y_predicted_regression = y_predicted_regression, y_true_minibatch = y_true_minibatch, y_true = y_true)
        progbar.update(current = i, values = [ 
          ("rpn_total_loss", stats.rpn_mean_total_loss),
          ("rpn_class_loss", stats.rpn_mean_class_loss),
          ("rpn_regression_loss", stats.rpn_mean_regression_loss),
          ("rpn_class_accuracy", stats.rpn_mean_class_accuracy),
          ("rpn_class_recall", stats.rpn_mean_class_recall) 
        ])
        stats.on_step_end()

      # Checkpoint
      print("")
      checkpoint_filename = "checkpoint-%d-%1.2f.hdf5" % (epoch, stats.rpn_mean_total_loss)
      rpn_model.save_weights(filepath = checkpoint_filename, overwrite = True, save_format = "h5")
      print("Saved checkpoint: %s" % checkpoint_filename)
      stats.on_epoch_end()

    # Save learned model parameters
    if options.save_to is not None:
      rpn_model.save_weights(filepath = options.save_to, overwrite = True, save_format = "h5")
      print("Saved model weights to %s" % options.save_to)
