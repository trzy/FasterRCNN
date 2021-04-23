# TODO:
#
# - Maximum number of proposals in the VOC dataset in a single image is only 157 when training from scratch (and this number drops further
#   as training progresses)
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

    rpn_total_losses = np.zeros(num_samples)
    class_losses = np.zeros(num_samples)
    regression_losses = np.zeros(num_samples)
    class_accuracies = np.zeros(num_samples)
    class_recalls = np.zeros(num_samples)

    for epoch in range(options.epochs):
      progbar = tf.keras.utils.Progbar(num_samples)
      print("Epoch %d/%d" % (epoch + 1, options.epochs))

      for i in range(num_samples):
        # Fetch one sample and reshape to batch size of 1
        # TODO: should we just return y_true_complete with a y_batch/y_valid?
        image_path, x, y, anchor_boxes = next(train_data)
        input_image_shape = x.shape
        y_true_complete = voc.get_image_description(image_path).get_complete_ground_truth_regressions_map()
        y_true_complete = y_true_complete.reshape((1, y_true_complete.shape[0], y_true_complete.shape[1], y_true_complete.shape[2], y_true_complete.shape[3]))
        y = y.reshape((1, y.shape[0], y.shape[1], y.shape[2], y.shape[3]))  # convert to batch size of 1
        x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))

        # RPN: back prop one step
        losses = rpn_model.train_on_batch(x = x, y = y) # loss = [sum, loss_cls, loss_regr]

        # RPN: predict so we can compute current accuracy
        y_predicted_class, y_predicted_regression = rpn_model.predict_on_batch(x = x)
        y_true_class = y_true_complete[:,:,:,:,2].reshape(y_predicted_class.shape)  # ground truth classes
        y_valid = y[:,:,:,:,0].reshape(y_predicted_class.shape)                     # valid anchors
        assert np.size(y_true_class) == np.size(y_predicted_class)

        # Extract proposals and convert to RPN space
        #TODO: we could also convert the anchors to feature map space and then, because regressed box parameters are expressed relative to anchor
        #      dimensions. Parameters converted to absolute values would then be in feature map space.
        proposals = region_proposal_network.extract_proposals(y_predicted_class = y_predicted_class, y_predicted_regression = y_predicted_regression, y_true = y, anchor_boxes = anchor_boxes)
        proposals = proposals[:,0:4]  # strip out class
        proposals = region_proposal_network.convert_box_coordinates_from_image_to_rpn_layer_space(box = proposals)

        # Perform clipping in RPN map space so that RoI pooling layer is never
        # passed rectangles that exceed the boundaries of its input map
        rpn_map_shape = region_proposal_network.compute_anchor_map_shape(input_image_shape = input_image_shape)
        proposals = region_proposal_network.clip_box_coordinates_to_map_boundaries(boxes = proposals, map_shape = rpn_map_shape)
        proposals = proposals.reshape((1, proposals.shape[0], proposals.shape[1]))  # batch size of 1

        # Test: run classifier model forward (not yet complete)
        if proposals.shape[1] > 0:
          proposals[:,:,2] = proposals[:,:,2] - proposals[:,:,0] + 1
          proposals[:,:,3] = proposals[:,:,3] - proposals[:,:,1] + 1
          y_final = classifier_model.predict_on_batch(x = [ x, proposals ])
          #print(y_final.shape)
          #exit()

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
        rpn_total_losses[i] = losses[0]
        class_losses[i] = losses[1]
        regression_losses[i] = losses[2]
        class_accuracies[i] = class_accuracy
        class_recalls[i] = class_recall
        mean_class_loss = np.mean(class_losses[0:i+1])
        mean_class_recall = np.mean(class_recalls[0:i+1])
        mean_regression_loss = np.mean(regression_losses[0:i+1])
        mean_rpn_total_loss = mean_class_loss + mean_regression_loss
        mean_class_accuracy = np.mean(class_accuracies[0:i+1])
        progbar.update(current = i, values = [ ("rpn_total_loss", mean_rpn_total_loss), ("class_loss", mean_class_loss), ("regression_loss", mean_regression_loss), ("class_accuracy", mean_class_accuracy), ("class_recall", mean_class_recall) ])

      # Checkpoint
      print("")
      checkpoint_filename = "checkpoint-%d-%1.2f.hdf5" % (epoch, mean_rpn_total_loss)
      rpn_model.save_weights(filepath = checkpoint_filename, overwrite = True, save_format = "h5")
      print("Saved checkpoint: %s" % checkpoint_filename)

    # Save learned model parameters
    if options.save_to is not None:
      rpn_model.save_weights(filepath = options.save_to, overwrite = True, save_format = "h5")
      print("Saved model weights to %s" % options.save_to)
