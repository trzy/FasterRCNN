# TODO: use lower IoU threshold of 0.1 for negative samples to perform hard sample mining (favor "difficult" samples that have some overlap with a positive class)

# TODO: try to run w/ 4 proposals and no L2 or dropout
# TODO: try to run w/ 64 proposals and compare difference
# TODO: try to freeze VGG layers?
# TODO: try to freeze layers and apply dropout?

#
# FasterRCNN for Keras
# Copyright 2021 Bart Trzynadlowski
#
# __main__.py
#
# Main module.
#

# TODO:
# - Standardize on notation for y_true maps
# - Desperately need to settle on some better naming conventions for the various y outputs and ground truths, as well as proposal
#   maps in different formats (e.g., pixel units, map units, etc.)
# - Clip final boxes? See: 2010_004041.jpg
# - Comment every non-trivial function and reformat code to 4-space tabs

# TODO:
#
# - Test whether K.abs()/tf.abs() fail on Linux
# - Weight decay, dropout, momentum

#
# Faster R-CNN in Keras: https://towardsdatascience.com/faster-r-cnn-object-detection-implemented-by-keras-for-custom-data-from-googles-open-images-125f62b9141a
# Understanding RoI pooling: https://towardsdatascience.com/understanding-region-of-interest-part-1-roi-pooling-e4f5dd65bb44
# NMS for object detection: https://towardsdatascience.com/non-maximum-suppression-nms-93ce178e177c
#

from . import utils
from . import visualization
from .dataset import VOC
from .models import vgg16
from .models import region_proposal_network
from .models.losses import rpn_class_loss
from .models.losses import rpn_regression_loss
from .models.losses import classifier_class_loss
from .models.losses import classifier_regression_loss
from .models import classifier_network
from .models.nms import nms
from .statistics import AveragePrecision
from .statistics import ModelStatistics

import argparse
from collections import defaultdict
import numpy as np
from math import exp
import os
import random
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import Model
from tensorflow.keras import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
import time


def prepare_mAP_directories():
  gt_dir = os.path.join(options.map_results, "ground-truth")
  results_dir = os.path.join(options.map_results, "detection-results")
  for dir in [ gt_dir, results_dir ]:
    if not os.path.exists(dir):
      os.makedirs(dir)

def write_results_to_disk(image_number, ground_truth_object_boxes, scored_boxes_by_class_index, voc):
  gt_file = os.path.join(options.map_results, "ground-truth", "image_%d.txt" % image_number)
  results_file = os.path.join(options.map_results, "detection-results", "image_%d.txt" % image_number)
  with open(gt_file, "w") as fp:
    for box in ground_truth_object_boxes:
      y_min, x_min, y_max, x_max = box.corners
      class_name = voc.index_to_class_name[box.class_index]
      fp.write("%s %f %f %f %f\n" % (class_name, x_min, y_min, x_max, y_max))
  with open(results_file, "w") as fp:
    for class_index, boxes in scored_boxes_by_class_index.items():
      class_name = voc.index_to_class_name[class_index]
      for box in boxes:
        y_min, x_min, y_max, x_max, score = box
        fp.write("%s %f %f %f %f %f\n" % (class_name, score, x_min, y_min, x_max, y_max))

def load_image(url, min_dimension_pixels, voc = None):
  """
  Loads image and returns NumPy tensor of shape (height,width,3). Image data is
  pre-processed for VGG-16.
  
  Parameters
  ----------
    url : str
      File to load. May be a file name or URL. If 'voc' dataset is provided,
      will first attempt to interpret 'url' as a filename from the dataset
      before falling back to treating it as a true URL.
    min_dimension_pixels: int
      New size of the image's minimum dimension. The other dimension will be
      scaled proportionally.
    voc : dataset.VOC, optional
      VOC dataset. If provided, allows files from the training and validation
      sets to be loaded by name only (rather than full path).

  Returns
  -------
  np.ndarray, PIL.Image 
    Image data as a NumPy tensor and PIL object.
  """
  if voc is not None:
    path = voc.get_full_path(filename = url)
    if os.path.exists(path):
      url = path
  return utils.load_image(url = url, min_dimension_pixels = min_dimension_pixels), utils.load_image_data_vgg16(url = url, min_dimension_pixels = min_dimension_pixels)

def print_weights(model):
  for layer in model.layers:
    weights = layer.get_weights()
    if len(weights) > 0:
      print(layer.name, layer.get_weights()[0][0])

def build_rpn_model(learning_rate, clipnorm, input_image_shape = (None, None, 3), weights_filepath = None, l2 = 0, freeze_layers = ""):
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

  # Freeze specified layers
  utils.freeze_layers(model = model, layers = freeze_layers)
  return model, conv_model

def build_classifier_model(num_classes, conv_model, learning_rate, clipnorm, dropout_fraction, l2 = 0, weights_filepath = None, freeze_layers = ""):
  proposal_boxes = Input(shape = (None, 4), dtype = tf.int32)
  classifier_output, regression_output = classifier_network.layers(num_classes = num_classes, input_map = conv_model.outputs[0], proposal_boxes = proposal_boxes, dropout_fraction = dropout_fraction, l2 = l2)
  model = Model([conv_model.input, proposal_boxes], [classifier_output, regression_output])

  optimizer = SGD(lr = learning_rate, momentum = 0.9, clipnorm = clipnorm)
  loss = [ classifier_class_loss, classifier_regression_loss ]
  model.compile(optimizer = optimizer, loss = loss)

  # Load weights
  if weights_filepath:
    model.load_weights(filepath = weights_filepath, by_name = True)
    print("Loaded classifier model weights from %s" % weights_filepath)

  # Freeze specified layers
  utils.freeze_layers(model = model, layers = freeze_layers)
  return model

def build_complete_model(rpn_model, classifier_model):
  model = Model(classifier_model.inputs, rpn_model.outputs + classifier_model.outputs)
  model.compile(optimizer = SGD(), loss = "mae")
  return model

def show_image(voc, filename):
  info = voc.get_image_info(path = voc.get_full_path(filename))

  # Need to build the model for this image size in order to be able to visualize boxes correctly
  conv_model = vgg16.conv_layers(input_shape = (info.height,info.width,3))
  classifier_output, regression_output = region_proposal_network.layers(input_map = conv_model.outputs[0])
  model = Model([conv_model.input], [classifier_output, regression_output])

  visualization.show_annotated_image(voc = voc, filename = options.show_image, draw_anchor_intersections = True, image_input_map = model.input, anchor_map = classifier_output)

def infer_rpn_boxes(rpn_model, voc, filename):
  """
  Run RPN model to find objects and draw their bounding boxes.
  """
  from .models.losses import rpn_class_loss_np
  info = voc.get_image_info(path = voc.get_full_path(filename))
  x = info.load_image_data()
  x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))
  y_class, y_regression = rpn_model.predict(x)
  for yy in range(y_class.shape[1]):
    for xx in range(y_class.shape[2]):
      for kk in range(y_class.shape[3]):
        if y_class[0,yy,xx,kk] > 0.5:
          print("%d,%d,%d -> %f" % (yy, xx, kk, y_class[0,yy,xx,kk]))
  y_true = info.get_ground_truth_map()
  y_true = y_true.reshape((1, y_true.shape[0], y_true.shape[1], y_true.shape[2], y_true.shape[3]))
  print("class loss=", rpn_class_loss_np(y_true=y_true, y_predicted=y_class))
  visualization.show_proposed_regions(voc = voc, filename = filename, y_true = y_true, y_class = y_class, y_regression = y_regression)

def filter_detections(proposals, classes, regressions, iou_threshold = 0.5):
  """
  Given proposals (in input image space) and the final predictions from the
  classifier network, returns a final set of boxes and their confidence scores 
  organized by class name. Redundant predictions are removed using non-maximum
  suppression. The results can directly be visualized atop the input image.

  Parameters
  ----------
    proposals : np.ndarray
      An (N,5) tensor of object proposals from the RPN network, where each 
      proposal is of the form (y_min, x_min, y_max, x_max, score). Coordinates
      are in input image (pixel) space and the score is the objectness class
      score.
    classes : np.ndarray
      An (N,num_classes) tensor of class predictions for each proposal. Classes
      are one-hot encoded with 0 being the background class.
    regressions : np.ndarray
      An (N,(num_classes-1)*4) tensor of predicted box regression parameters
      for the non-background classes. Index 0 corresponds to class index 1,
      index 4 to class index 2, etc. There are 4 parameters (ty, tx, th, tw)
      for each class. Only the 4 parameters for the predicted class (the
      highest scoring class index in the equivalent row of 'classes') are
      valid.
    iou_threshold : float
      IoU threshold for non-maximum supression. Used to remove redundant
      predictions.
    
  Returns
  -------
  dict
  Lists of boxes and their score, (y_min, x_min, y_max, x_max, score), by class
  index.
  """
  # Inputs must all be a single sample (no batches)
  assert len(classes.shape) == 2
  assert len(regressions.shape) == 2
  assert classes.shape[0] == regressions.shape[0]
  assert classes.shape[0] == proposals.shape[0]

  # Separate out results per class: class_name -> (y1, x1, y2, x2, score)
  result_by_class_idx = defaultdict(list)
  for i in range(classes.shape[0]):
    class_idx = np.argmax(classes[i,:])
    if class_idx > 0:
      regression_idx = (class_idx - 1) * 4
      box_params = regressions[i, regression_idx+0 : regression_idx+4]
      proposal_center_y = 0.5 * (proposals[i,0] + proposals[i,2])
      proposal_center_x = 0.5 * (proposals[i,1] + proposals[i,3])
      proposal_height = proposals[i,2] - proposals[i,0] + 1
      proposal_width = proposals[i,3] - proposals[i,1] + 1
      y1, x1, y2, x2 = region_proposal_network.convert_parameterized_box_to_points(box_params = box_params, anchor_center_y = proposal_center_y, anchor_center_x = proposal_center_x, anchor_height = proposal_height, anchor_width = proposal_width)
      result_by_class_idx[class_idx].append((y1, x1, y2, x2, classes[i,class_idx]))

  # Perform NMS for each class
  scored_boxes_by_class_idx = {}
  for class_idx, results in result_by_class_idx.items():
    results = np.vstack(results)
    indices = nms(proposals = results, iou_threshold = iou_threshold)
    results = results[indices]
    scored_boxes_by_class_idx[class_idx] = results

  return scored_boxes_by_class_idx

def show_objects(rpn_model, classifier_model, image, image_data):
  # TODO: streamline further and clean up
  x = image_data
  input_image_shape = x.shape
  rpn_shape = vgg16.compute_output_map_shape(input_image_shape = input_image_shape)
  anchor_boxes, anchor_boxes_valid = region_proposal_network.compute_all_anchor_boxes(input_image_shape = input_image_shape)
  x = np.expand_dims(x, axis = 0)
  anchor_boxes_valid = np.expand_dims(anchor_boxes_valid, axis = 0)
  y_rpn_predicted_class, y_rpn_predicted_regression = rpn_model.predict(x)
  proposals_pixels = region_proposal_network.extract_proposals(y_predicted_class = y_rpn_predicted_class, y_predicted_regression = y_rpn_predicted_regression, input_image_shape = input_image_shape, anchor_boxes = anchor_boxes, anchor_boxes_valid = anchor_boxes_valid)
  if proposals_pixels.shape[0] > 0:
    proposals = convert_proposals_to_classifier_network_format(proposals = proposals_pixels, rpn_shape = rpn_shape)
    proposals = np.expand_dims(proposals, axis = 0)
    # Run prediction
    y_classifier_predicted_class, y_classifier_predicted_regression = classifier_model.predict_on_batch(x = [ x, proposals ])
    # Filter the results by performing NMS per class and returning final boxes by class name 
    scored_boxes_by_class_idx = filter_detections(proposals = proposals_pixels, classes = y_classifier_predicted_class[0,:,:], regressions = y_classifier_predicted_regression[0,:,:])
    scored_boxes_by_class_name = { voc.index_to_class_name[class_idx]: boxes for class_idx, boxes in scored_boxes_by_class_idx.items() }
    for class_name, boxes in scored_boxes_by_class_name.items():
      for box in boxes:
        print("%s -> %d, %d, %d, %d (score=%f)" % (class_name, round(box[0]), round(box[1]), round(box[2]), round(box[3]), box[4]))
    # Show objects
    visualization.show_objects(image = image, boxes_by_class_name = scored_boxes_by_class_name)
  else:
    print("No proposals generated")

def sample_proposals(proposals, y_true_proposal_classes, y_true_proposal_regressions, max_proposals, positive_fraction):
  if max_proposals <= 0:
    return proposals, y_true_proposal_classes, y_true_proposal_regressions

  # Get positive and negative (background) proposals
  class_indices = np.argmax(y_true_proposal_classes, axis = 1)  # [N,num_classes] -> [N], where each element is the class index (highest score from its row)
  positive_indices = np.argwhere(class_indices > 0)[:,0]
  negative_indices = np.argwhere(class_indices <= 0)[:,0]
  num_positive_proposals = len(positive_indices)
  num_negative_proposals = len(negative_indices)
  
  # Select positive and negative samples, if there are enough
  num_samples = min(max_proposals, len(class_indices))
  num_positive_samples = min(round(num_samples * positive_fraction), num_positive_proposals)
  num_negative_samples = min(num_samples - num_positive_samples, num_negative_proposals)

  # Do we have enough?
  if num_positive_samples <= 0 or num_negative_samples <= 0:
    return proposals[[]], y_true_proposal_classes[[]], y_true_proposal_regressions[[]]  # return 0-length tensors

  # Sample randomly
  positive_sample_indices = np.random.choice(positive_indices, size = num_positive_samples, replace = False)
  negative_sample_indices = np.random.choice(negative_indices, size = num_negative_samples, replace = False)
  indices = np.concatenate([ positive_sample_indices, negative_sample_indices ])

  # Return
  return proposals[indices], y_true_proposal_classes[indices], y_true_proposal_regressions[indices]
   
def select_proposals_for_training(proposals, ground_truth_object_boxes, num_classes, max_proposals, positive_fraction):
  """
  Given proposals generated by the RPN, selects a random subset for training
  the classifier network and generates the corresponding ground truth maps
  (object class and bounding box regression targets).
 
  Parameters
  ----------
    proposals : np.ndarray
      List of object proposals obtained from forward pass of RPN model, with
      shape (N,4). Proposals are box coordinates in image space: (y_min, x_min,
      y_max, x_max).
    ground_truth_object_boxes : list(VOC.Box)
      A list of ground truth box data.
    num_classes : int
      Total number of object classes in classifier model, including background
      class 0.
    max_proposals : int
      Maximum number of proposals to use for training. If <= 0, all proposals
      are used.
    positive_fraction : int
      Desired fraction of positive proposals. Determines the maximimum number
      of positive samples to use. Less may be used.

  Returns
  -------
  np.ndarray, np.ndarray, np.ndarray
    A tuple of 3 maps with N rows each. Each row in a given map corresponds to
    one of the N proposals.
    1. Proposals, shape (N,4), where N <= max_proposals if max_proposals given.
    2. Ground truth proposal classes, shape (N,num_classes), where each class
       is one-hot encoded.
    3. Ground truth proposal regression targets, shape (N,(num_classes-1)*4).
       Each row contains parameterized regression targets for each possible
       non-background class (that is, (N,0:4) corresponds to class 1). Only the
       4 values corresponding to the ground truth class are valid and the rest
       of the row will be 0. For example:
        classes = [ 0, 0, 1, 0, ..., 0 ]
        regressions = [ 0, 0, 0, 0, ty, tx, th, tw, 0, 0, 0, 0, ... 0 ]
  """
  # Generate one-hot labels for each proposal
  proposals, y_true_proposal_classes, y_true_proposal_regressions = region_proposal_network.label_proposals(
    proposals = proposals,
    ground_truth_object_boxes = ground_truth_object_boxes,
    num_classes = num_classes,
    min_iou_threshold = options.min_iou,
    max_iou_threshold = options.max_iou)

  # Sample from proposals
  proposals, y_true_proposal_classes, y_true_proposal_regressions = sample_proposals(
    proposals = proposals,
    y_true_proposal_classes = y_true_proposal_classes,
    y_true_proposal_regressions = y_true_proposal_regressions,
    max_proposals = max_proposals,
    positive_fraction = positive_fraction)

  return proposals, y_true_proposal_classes, y_true_proposal_regressions

def convert_proposals_to_classifier_network_format(proposals, rpn_shape):
  """
  Converts a series of proposals in input image space to the form expected by
  the classifier network. The box coordinates are converted to anchor space
  (RPN output shape) and their format is converted from top-left and bottom-
  right points to top-left point and shape, as expected by the RoI pooling
  layer that is the first layer in the classifier.

  Parameters
  ----------
    proposals : np.ndarray
      Proposals with shape (N,5), as box coordinates in image space and an
      objectness score: (y_min, x_min, y_max, x_max, score).
    rpn_shape : (int, int, int)
      The shape of the output map of the convolutional network stage and the
      input to the RPN.

  Returns
  -------
  np.ndarray
    A map of shape (N,4) in the format expected by the  RoI pooling layer,
    where each box is now: (y_min, x_min, height, width), in RPN map units.
    The score is stripped out.
  """
  # Strip out score leaving only the box coordinates
  proposals = proposals[:,0:4]

  # Convert to anchor map (RPN map) space
  boxes = vgg16.convert_box_coordinates_from_image_to_output_map_space(box = proposals, output_map_shape = rpn_shape)

  # Convert from (y_min,x_min,y_max,x_max) -> (y_min,x_min,height,width) as expected by RoI pool layer
  boxes[:,2:4] = boxes[:,2:4] - boxes[:,0:2] + 1

  # RoI pooling layer expects tf.int32
  rois = boxes.astype(np.int32)
  return rois
  
def validate(rpn_model, classifier_model, voc):
  val_data = voc.validation_data(mini_batch_size = options.mini_batch)
  num_samples = voc.num_samples["val"]

  mAP = AveragePrecision()
  stats = ModelStatistics(num_samples = num_samples)
  stats.on_epoch_begin()
  progbar = tf.keras.utils.Progbar(num_samples)
  
  for i in range(num_samples):
    stats.on_step_begin()

    # Fetch next sample
    image_path, x, y_true_minibatch, y_true, anchor_boxes, ground_truth_object_boxes = next(val_data)
    input_image_shape = x.shape
    rpn_shape = vgg16.compute_output_map_shape(input_image_shape = input_image_shape)
    anchor_boxes, anchor_boxes_valid = region_proposal_network.compute_all_anchor_boxes(input_image_shape = input_image_shape)
    x = np.expand_dims(x, axis = 0)
    y_true_minibatch = np.expand_dims(y_true_minibatch, axis = 0)
    anchor_boxes_valid = np.expand_dims(anchor_boxes_valid, axis = 0)
    y_true = np.expand_dims(y_true, axis = 0)

    # RPN prediction step and evaluation
    y_rpn_predicted_class, y_rpn_predicted_regression = rpn_model.predict(x)
    rpn_losses = rpn_model.evaluate(x = x, y = y_true_minibatch, verbose = False)

    # Extract proposals
    proposals = region_proposal_network.extract_proposals(
      y_predicted_class = y_rpn_predicted_class, 
      y_predicted_regression = y_rpn_predicted_regression, 
      input_image_shape = input_image_shape, 
      anchor_boxes = anchor_boxes, 
      anchor_boxes_valid = anchor_boxes_valid)

    scored_boxes_by_class_index = {}

    if proposals.shape[0] > 0:
      # Get proposal labels as we would do during training but try to use all
      # proposals
      proposals, y_true_proposal_classes, y_true_proposal_regressions = select_proposals_for_training(
        proposals = proposals,
        ground_truth_object_boxes = ground_truth_object_boxes,
        num_classes = voc.num_classes,
        max_proposals = 0,  # use all of them
        positive_fraction = 0.5
      )
      proposals_pixels = proposals  # save proposals in input image space
      proposals = convert_proposals_to_classifier_network_format(proposals = proposals, rpn_shape = rpn_shape)
      proposals = np.expand_dims(proposals, axis = 0)
      y_true_proposal_classes = np.expand_dims(y_true_proposal_classes, axis = 0)
      y_true_proposal_regressions = np.expand_dims(y_true_proposal_regressions, axis = 0)

      # Classifier prediction step
      y_classifier_predicted_class, y_classifier_predicted_regression = classifier_model.predict_on_batch(x = [ x, proposals ])
      classifier_losses = classifier_model.evaluate(x = [ x, proposals ], y = [ y_true_proposal_classes, y_true_proposal_regressions ], verbose = False)
      scored_boxes_by_class_index = filter_detections(
        proposals = proposals_pixels,
        classes = y_classifier_predicted_class[0,:,:],
        regressions = y_classifier_predicted_regression[0,:,:])

      # Write detections out to disk for off-line mAP calculation
      if options.map_results:
        write_results_to_disk(
          image_number = i,
          ground_truth_object_boxes = ground_truth_object_boxes,
          scored_boxes_by_class_index = scored_boxes_by_class_index,
          voc = voc)
      
      # Update classifier stats
      stats.on_classifier_step(
        losses = classifier_losses,
        y_predicted_class = y_classifier_predicted_class,
        y_predicted_regression = y_classifier_predicted_regression,
        y_true_classes = y_true_proposal_classes,
        y_true_regressions = y_true_proposal_regressions,
        timing_samples = {}
      )

      del proposals_pixels
      del y_classifier_predicted_class
      del y_classifier_predicted_regression
      del y_true_proposal_classes
      del y_true_proposal_regressions

    # Update mAP calculation
    mAP.add_image_results(scored_boxes_by_class_index = scored_boxes_by_class_index, ground_truth_object_boxes = ground_truth_object_boxes)

    # Update RPN progress and progress bar
    stats.on_rpn_step(
      losses = rpn_losses,
      y_predicted_class = y_rpn_predicted_class,
      y_predicted_regression = y_rpn_predicted_regression,
      y_true_minibatch = y_true_minibatch,
      y_true = y_true,
      timing_samples = {}
    )
  
    progbar.update(current = i, values = [ 
      ("rpn_total_loss", stats.rpn_mean_total_loss),
      ("rpn_class_loss", stats.rpn_mean_class_loss),
      ("rpn_regression_loss", stats.rpn_mean_regression_loss),
      ("rpn_class_accuracy", stats.rpn_class_accuracy),
      ("rpn_class_recall", stats.rpn_class_recall),
      ("classifier_total_loss", stats.classifier_mean_total_loss),
      ("classifier_class_loss", stats.classifier_mean_class_loss),
      ("classifier_regression_loss", stats.classifier_mean_regression_loss)
    ])
    stats.on_step_end()

    # Try hinting to interpreter that we are finished with these objects 
    del proposals
    del x
    del y_true_minibatch
    del y_true
    del scored_boxes_by_class_index
    del anchor_boxes
    del ground_truth_object_boxes
    del anchor_boxes_valid

  stats.on_epoch_end()
  
  # Print mAP
  print("Mean Average Precision: %1.2f%%" % (100.0 * mAP.compute_mean_average_precision(interpolated = True)))

def train(rpn_model, classifier_model, voc):
  train_data = voc.train_data(mini_batch_size = options.mini_batch, augment = options.augment, cache_images = True)
  num_samples = voc.num_samples["train"]  # number of iterations in an epoch

  stats = ModelStatistics(num_samples = num_samples)
  logger = utils.CSVLogCallback(filename = options.log, log_epoch_number = False, log_learning_rate = False)

  for epoch in range(options.epochs):
    stats.on_epoch_begin()
    progbar = tf.keras.utils.Progbar(num_samples)
    print("Epoch %d/%d" % (epoch + 1, options.epochs))

    for i in range(num_samples):
      stats.on_step_begin()

      # Fetch one sample and reshape to batch size of 1
      rpn_train_t0 = time.perf_counter()
      image_path, x, y_true_minibatch, y_true, anchor_boxes, ground_truth_object_boxes = next(train_data)
      input_image_shape = x.shape
      rpn_shape = vgg16.compute_output_map_shape(input_image_shape = input_image_shape)
      y_true = np.expand_dims(y_true, axis = 0)
      y_true_minibatch = np.expand_dims(y_true_minibatch, axis = 0)
      x = np.expand_dims(x, axis = 0)

      # RPN: back prop one step (and then predict so we can evaluate accuracy)
      rpn_losses = rpn_model.train_on_batch(x = x, y = y_true_minibatch) # loss = [sum, loss_cls, loss_regr]
      y_rpn_predicted_class, y_rpn_predicted_regression = rpn_model.predict_on_batch(x = x)
      rpn_train_time = time.perf_counter() - rpn_train_t0

      # Classifier
      if not options.rpn_only:
        extract_proposals_t0 = time.perf_counter()
        proposals = region_proposal_network.extract_proposals(
          y_predicted_class = y_rpn_predicted_class,
          y_predicted_regression = y_rpn_predicted_regression,
          input_image_shape = input_image_shape,
          anchor_boxes = anchor_boxes,
          anchor_boxes_valid = y_true[:,:,:,:,0],
          max_proposals = options.max_proposals
        )
        extract_proposals_time = time.perf_counter() - extract_proposals_t0

        if proposals.shape[0] > 0:
          # Prepare proposals and ground truth data for classifier network 
          prepare_proposals_t0 = time.perf_counter()
          proposals, y_true_proposal_classes, y_true_proposal_regressions = select_proposals_for_training(
            proposals = proposals,
            ground_truth_object_boxes = ground_truth_object_boxes,
            num_classes = voc.num_classes,
            max_proposals = options.proposal_batch,
            positive_fraction = 0.5
          )
          proposals = convert_proposals_to_classifier_network_format(
            proposals = proposals,
            rpn_shape = rpn_shape
          )
          proposals = np.expand_dims(proposals, axis = 0)
          y_true_proposal_classes = np.expand_dims(y_true_proposal_classes, axis = 0)
          y_true_proposal_regressions = np.expand_dims(y_true_proposal_regressions, axis = 0)
          prepare_proposals_time = time.perf_counter() - prepare_proposals_t0

          # Do we have any proposals to process?
          if proposals.size > 0:
            """
            get_conv_output = K.function([rpn_model.layers[0].input],[rpn_model.get_layer("block5_conv3").output])
            conv_output = get_conv_output([x])[0]

            get_pool_output = K.function(classifier_model.inputs,[classifier_model.get_layer("roi_pool").output])
            pool_output = get_pool_output([x, proposals])[0]
            debug_pool_layer(feature_map = conv_output, rois = proposals, pool_output = pool_output)
            """

            # Classifier: back prop one step (and then predict)
            classifier_train_t0 = time.perf_counter()
            classifier_losses = classifier_model.train_on_batch(x = [ x, proposals ], y = [ y_true_proposal_classes, y_true_proposal_regressions ])
            y_classifier_predicted_class, y_classifier_predicted_regression = classifier_model.predict_on_batch(x = [ x, proposals ])
            classifier_train_time = time.perf_counter() - classifier_train_t0

            # Update classifier progress
            stats.on_classifier_step(
              losses = classifier_losses,
              y_predicted_class = y_classifier_predicted_class,
              y_predicted_regression = y_classifier_predicted_regression,
              y_true_classes = y_true_proposal_classes,
              y_true_regressions = y_true_proposal_regressions,
              timing_samples = { "extract_proposals": extract_proposals_time, "prepare_proposals": prepare_proposals_time, "classifier_train": classifier_train_time }
            )

            del y_classifier_predicted_class
            del y_classifier_predicted_regression

          del y_true_proposal_classes
          del y_true_proposal_regressions

        del proposals

      # Update RPN progress and progress bar
      stats.on_rpn_step(
        losses = rpn_losses,
        y_predicted_class = y_rpn_predicted_class,
        y_predicted_regression = y_rpn_predicted_regression,
        y_true_minibatch = y_true_minibatch,
        y_true = y_true,
        timing_samples = { "rpn_train": rpn_train_time }
      )
      progbar.update(current = i, values = [ 
        ("rpn_total_loss", stats.rpn_mean_total_loss),
        ("rpn_class_loss", stats.rpn_mean_class_loss),
        ("rpn_regression_loss", stats.rpn_mean_regression_loss),
        ("rpn_class_accuracy", stats.rpn_class_accuracy),
        ("rpn_class_recall", stats.rpn_class_recall),
        ("classifier_total_loss", stats.classifier_mean_total_loss),
        ("classifier_class_loss", stats.classifier_mean_class_loss),
        ("classifier_regression_loss", stats.classifier_mean_regression_loss)
      ])
      stats.on_step_end()
      
      del y_rpn_predicted_class
      del y_rpn_predicted_regression
      del y_true_minibatch
      del y_true
      del anchor_boxes
      del ground_truth_object_boxes
      del x      

    # Log
    logger.on_epoch_end(epoch = epoch, logs = {
      "epoch": epoch,
      "learning_rate": options.learning_rate,
      "clipnorm": options.clipnorm,
      "mini_batch": options.mini_batch,
      "max_proposals": options.max_proposals,
      "proposal_batch": options.proposal_batch,
      "rpn_total_loss": stats.rpn_mean_total_loss,
      "rpn_class_loss": stats.rpn_mean_class_loss,
      "rpn_regression_loss": stats.rpn_mean_regression_loss,
      "rpn_class_accuracy": stats.rpn_class_accuracy,
      "rpn_class_recall": stats.rpn_class_recall,
      "classifier_total_loss": stats.classifier_mean_total_loss,
      "classifier_class_loss": stats.classifier_mean_class_loss,
      "classifier_regression_loss": stats.classifier_mean_regression_loss
    })

    # Checkpoint
    print("")
    checkpoint_filename = "checkpoint-%d-%1.2f.hdf5" % (epoch, stats.rpn_mean_total_loss)
    complete_model.save_weights(filepath = checkpoint_filename, overwrite = True, save_format = "h5")
    print("Saved checkpoint: %s" % checkpoint_filename)
    stats.on_epoch_end()
 
  # Save learned model parameters
  if options.save_to is not None:
    complete_model.save_weights(filepath = options.save_to, overwrite = True, save_format = "h5")
    print("Saved model weights to %s" % options.save_to)


# good test images:
# 2010_004041.jpg
# 2010_005080.jpg
# 2008_000019.jpg
# 2009_004872.jpg
if __name__ == "__main__":
  parser = argparse.ArgumentParser("FasterRCNN")
  parser.add_argument("--dataset-dir", metavar = "path", type = str, action = "store", default = "../VOCdevkit/VOC2012", help = "Dataset directory")
  parser.add_argument("--train-dataset", metavar = "name", type = str, action = "store", default = "train", help = "Training dataset to use (train or trainval)")
  parser.add_argument("--val-dataset", metavar = "name", type = str, action = "store", default = "val", help = "Validation dataset to use (val or test)")
  parser.add_argument("--show-image", metavar = "file", type = str, action = "store", help = "Show an image with ground truth and corresponding anchor boxes")
  parser.add_argument("--validate", action = "store_true", help = "Validate the model using the validation dataset")
  parser.add_argument("--map-results", metavar = "dir", type = str, action = "store", help = "Run validation and write out results for Cartucho mAP analysis to directory")
  parser.add_argument("--train", action = "store_true", help = "Train the model on the training dataset")
  parser.add_argument("--epochs", metavar = "count", type = utils.positive_int, action = "store", default = "10", help = "Number of epochs to train for")
  parser.add_argument("--learning-rate", metavar = "rate", type = float, action = "store", default = "0.001", help = "Learning rate")
  parser.add_argument("--clipnorm", metavar = "value", type = float, action = "store", default = "1.0", help = "Clip gradient norm to value")
  parser.add_argument("--mini-batch", metavar = "size", type = utils.positive_int, action = "store", default = 256, help = "Anchor mini-batch size (per image) for region proposal network")
  parser.add_argument("--max-proposals", metavar = "size", type = utils.positive_int, action = "store", default = 0, help = "Maximum number of proposals to extract")
  parser.add_argument("--proposal-batch", metavar = "size", type = utils.positive_int, action = "store", default = 4, help = "Proposal batch size (per image) for classifier network")
  parser.add_argument("--min-iou", metavar="value", type = float, default = "0", action = "store", help = "Minimum IoU for selection of negative RoI samples for input to classifier")
  parser.add_argument("--max-iou", metavar="value", type = float, default = "0.5", action = "store", help = "Maximum IoU for selection of negative RoI samples for input to classifier")
  parser.add_argument("--augment", action="store_true", help = "Augment training data with random horizontal flips")
  parser.add_argument("--l2", metavar = "value", type = float, action = "store", default = "2.5e-4", help = "L2 regularization")
  parser.add_argument("--dropout", metavar = "value", type = float, default = "0", action = "store", help = "Dropout fraction on last 2 fully connected layers")
  parser.add_argument("--freeze", metavar = "layers", action = "store", help = "Freeze the specified layers during training")
  parser.add_argument("--rpn-only", action = "store_true", help = "Train only the region proposal network")
  parser.add_argument("--log", metavar = "filepath", type = str, action = "store", default = "out.csv", help = "Log metrics to csv file")
  parser.add_argument("--save-to", metavar = "filepath", type = str, action = "store", help = "File to save model weights to when training is complete")
  parser.add_argument("--load-from", metavar="filepath", type = str, action = "store", help = "File to load initial model weights from")
  parser.add_argument("--infer-boxes", metavar = "file", type = str, action = "store", help = "Run inference on image using region proposal network and display bounding boxes")
  parser.add_argument("--show-objects", metavar = "file", type = str, action = "store", help = "Run inference on image using classifier network and display bounding boxes")
  options = parser.parse_args()

  voc = VOC(dataset_dir = options.dataset_dir, min_dimension_pixels = 600, train_dataset = options.train_dataset, val_dataset = options.val_dataset)

  rpn_model, conv_model = build_rpn_model(weights_filepath = options.load_from, learning_rate = options.learning_rate, clipnorm = options.clipnorm, l2 = options.l2, freeze_layers = options.freeze)
  classifier_model = build_classifier_model(num_classes = voc.num_classes, conv_model = conv_model, weights_filepath = options.load_from, learning_rate = options.learning_rate, clipnorm = options.clipnorm, dropout_fraction = options.dropout, l2 = options.l2, freeze_layers = options.freeze)
  complete_model = build_complete_model(rpn_model = rpn_model, classifier_model = classifier_model) # contains all weights, used for saving weights
  complete_model.summary()

  # Run-time environment
  cuda_available = tf.test.is_built_with_cuda()
  gpu_available = tf.test.is_gpu_available(cuda_only = False, min_cuda_compute_capability = None)
  print("CUDA Available : %s" % ("yes" if cuda_available else "no"))
  print("GPU Available  : %s" % ("yes" if gpu_available else "no"))
  print("Eager Execution: %s" % ("yes" if tf.executing_eagerly() else "no"))

  if options.show_image:
    show_image(voc = voc, filename = options.show_image)

  if options.infer_boxes:
    infer_rpn_boxes(rpn_model = rpn_model, voc = voc, filename = options.infer_boxes)

  if options.show_objects:
    image, image_data = load_image(url = options.show_objects, min_dimension_pixels = 600, voc = voc)
    show_objects(rpn_model = rpn_model, classifier_model = classifier_model, image = image, image_data = image_data)

  if options.train:
    train(rpn_model = rpn_model, classifier_model = classifier_model, voc = voc)

  if options.train or options.validate or options.map_results:  # always validate after training
    if options.map_results:
      prepare_mAP_directories()
    validate(rpn_model = rpn_model, classifier_model = classifier_model, voc = voc)
