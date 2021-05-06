# TODO:
# - Desperately need to return a separate map indicating anchor validity and then force it to be passed in explicitly, including
#   to training process, so that y_true becomes a tuple of two maps. 
# - Desperately need to settle on some better naming conventions for the various y outputs and ground truths, as well as proposal
#   maps in different formats (e.g., pixel units, map units, etc.)

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
from .models.losses import rpn_class_loss
from .models.losses import rpn_regression_loss
from .models.losses import classifier_class_loss
from .models.losses import classifier_regression_loss
from .models import classifier_network
from .models.nms import nms

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

def build_classifier_model(num_classes, conv_model, learning_rate, clipnorm, weights_filepath = None):
  proposal_boxes = Input(shape = (None, 4), dtype = tf.int32)
  classifier_output, regression_output = classifier_network.layers(num_classes = num_classes, input_map = conv_model.outputs[0], proposal_boxes = proposal_boxes)
  model = Model([conv_model.input, proposal_boxes], [classifier_output, regression_output])

  optimizer = SGD(lr = learning_rate, momentum = 0.9, clipnorm = clipnorm)
  loss = [ classifier_class_loss, classifier_regression_loss ]
  model.compile(optimizer = optimizer, loss = loss)

  # Load weights
  if weights_filepath:
    model.load_weights(filepath = weights_filepath, by_name = True)
    print("Loaded classifier model weights from %s" % weights_filepath)

  return model

def build_complete_model(rpn_model, classifier_model):
  model = Model(classifier_model.inputs, rpn_model.outputs + classifier_model.outputs)
  model.compile(optimizer = SGD(), loss = "mae")
  return model

def show_image(voc, filename):
  info = voc.get_image_description(path = voc.get_full_path(filename))

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
  info = voc.get_image_description(path = voc.get_full_path(filename))
  x = info.load_image_data()
  x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))
  y_class, y_regression = rpn_model.predict(x)
  for yy in range(y_class.shape[1]):
    for xx in range(y_class.shape[2]):
      for kk in range(y_class.shape[3]):
        if y_class[0,yy,xx,kk] > 0.5:
          print("%d,%d,%d -> %f" % (yy, xx, kk, y_class[0,yy,xx,kk]))
  y_true = info.get_complete_ground_truth_regressions_map()
  y_true = y_true.reshape((1, y_true.shape[0], y_true.shape[1], y_true.shape[2], y_true.shape[3]))
  print("class loss=", rpn_class_loss_np(y_true=y_true, y_predicted=y_class))
  visualization.show_proposed_regions(voc = voc, filename = filename, y_true = y_true, y_class = y_class, y_regression = y_regression)

def filter_classifier_results(proposals, classes, regressions, voc, iou_threshold = 0.5):

  # Inputs must all be a single sample (no batches)
  assert len(classes.shape) == 2
  assert len(regressions.shape) == 2
  assert classes.shape[0] == regressions.shape[0]
  assert classes.shape[0] == proposals.shape[0]

  # Separate out results per class: class_name -> (y1, x1, y2, x2, score)
  result_by_class_name = defaultdict(list)
  for i in range(classes.shape[0]):
    class_idx = np.argmax(classes[i,:])
    if class_idx > 0:
      class_name = voc.index_to_class_name[class_idx]
      regression_idx = (class_idx - 1) * 4
      box_params = regressions[i, regression_idx+0 : regression_idx+4]
      proposal_center_y = 0.5 * (proposals[i,0] + proposals[i,2])
      proposal_center_x = 0.5 * (proposals[i,1] + proposals[i,3])
      proposal_height = proposals[i,2] - proposals[i,0] + 1
      proposal_width = proposals[i,3] - proposals[i,1] + 1
      y1, x1, y2, x2 = region_proposal_network.convert_parameterized_box_to_points(box_params = box_params, anchor_center_y = proposal_center_y, anchor_center_x = proposal_center_x, anchor_height = proposal_height, anchor_width = proposal_width)
      result_by_class_name[class_name].append((y1, x1, y2, x2, classes[i,class_idx]))

  # Perform NMS for each class
  boxes_by_class_name = {}
  for class_name, results in result_by_class_name.items():
    results = np.vstack(results)
    indices = nms(proposals = results, iou_threshold = iou_threshold)
    results = results[indices]
    boxes_by_class_name[class_name] = results[:,0:4]  # strip off the score

  return boxes_by_class_name

def show_objects(rpn_model, classifier_model, voc, filename):
  # TODO: ugh, what a mess! This needs to be streamlined.
  # TODO: we need a way to get anchor boxes and the valid mask independently from y_true
  info = voc.get_image_description(path = voc.get_full_path(filename))
  x = info.load_image_data()
  y_rpn_true = info.get_complete_ground_truth_regressions_map()
  input_image_shape = x.shape
  cnn_output_shape = vgg16.compute_output_map_shape(input_image_shape = input_image_shape)
  anchor_boxes, anchor_boxes_valid = region_proposal_network.compute_all_anchor_boxes(input_image_shape = input_image_shape)
  x = np.expand_dims(x, axis = 0)
  y_rpn_true = np.expand_dims(y_rpn_true, axis = 0)
  y_rpn_class, y_rpn_regression = rpn_model.predict(x)
  proposals = region_proposal_network.extract_proposals(y_predicted_class = y_rpn_class, y_predicted_regression = y_rpn_regression, y_true = y_rpn_true, anchor_boxes = anchor_boxes)
  if proposals.shape[0] > 0:
    # TODO: convert_proposals_to_classifier_network_format() needs to be modified to work for inference, where ground truth is not needed
    proposals = proposals[:,0:4]
    proposals = region_proposal_network.clip_box_coordinates_to_map_boundaries(boxes = proposals, map_shape = input_image_shape)
    proposals_pixels = proposals
    proposals = vgg16.convert_box_coordinates_from_image_to_output_map_space(box = proposals, output_map_shape = cnn_output_shape)
    proposals[:,2:4] = proposals[:,2:4] - proposals[:,0:2] + 1
    proposals = proposals.astype(np.int32)
    proposals = np.expand_dims(proposals, axis = 0)
    # Run prediction
    y_classifier_predicted_class, y_classifier_predicted_regression = classifier_model.predict_on_batch(x = [ x, proposals ])
    # Filter the results by performing NMS per class and returning final boxes by class name 
    boxes_by_class_name = filter_classifier_results(proposals = proposals_pixels, classes = y_classifier_predicted_class[0,:,:], regressions = y_classifier_predicted_regression[0,:,:], voc = voc)
    for class_name, boxes in boxes_by_class_name.items():
      for box in boxes:
        print("%s -> %d, %d, %d, %d" % (class_name, round(box[0]), round(box[1]), round(box[2]), round(box[3])))
    # Show objects
    visualization.show_objects(voc = voc, filename = filename, boxes_by_class_name = boxes_by_class_name)
  else:
    print("No proposals generated")

class TrainingStatistics:
  def __init__(self):
    self._step_number = 0

    self._rpn_total_losses = np.zeros(num_samples)
    self._rpn_class_losses = np.zeros(num_samples)
    self._rpn_regression_losses = np.zeros(num_samples)
    self._rpn_class_accuracies = np.zeros(num_samples)
    self._rpn_class_recalls = np.zeros(num_samples)

    self._classifier_total_losses = np.zeros(num_samples)
    self._classifier_class_losses = np.zeros(num_samples)
    self._classifier_regression_losses = np.zeros(num_samples)

    self._rpn_regression_targets = np.zeros((0,4))
    self._classifier_regression_targets = np.zeros((0,4))
    self._classifier_regression_predictions = np.zeros((0,4))

    self.rpn_mean_class_loss = float("inf")
    self.rpn_mean_class_accuracy = 0
    self.rpn_mean_class_recall = 0
    self.rpn_mean_regression_loss = float("inf")
    self.rpn_mean_total_loss = float("inf")
    
    self.classifier_mean_class_loss = float("inf")
    self.classifier_mean_class_accuracy = 0
    self.classifier_mean_class_recall = 0
    self.classifier_mean_regression_loss = float("inf")
    self.classifier_mean_total_loss = float("inf")

  def on_epoch_begin(self):
    """
    Must be called at the beginning of each epoch.
    """
    self._step_number = 0
    self._rpn_regression_targets = np.zeros((0,4))
    self._classifier_regression_targets = np.zeros((0,4))
    self._classifier_regression_predictions = np.zeros((0,4))

  def on_epoch_end(self):
    """
    Must be called at the end of each epoch after the last step.
    """
    # Print stats for RPN regression targets
    mean_ty, mean_tx, mean_th, mean_tw = np.mean(self._rpn_regression_targets, axis = 0)
    std_ty, std_tx, std_th, std_tw = np.std(self._rpn_regression_targets, axis = 0)
    print("RPN Regression Target Means : %1.2f %1.2f %1.2f %1.2f" % (mean_ty, mean_tx, mean_th, mean_tw))
    print("RPN Regression Target StdDev: %1.2f %1.2f %1.2f %1.2f" % (std_ty, std_tx, std_th, std_tw))
    # Print stats for classifier regression targets
    mean_ty, mean_tx, mean_th, mean_tw = np.mean(self._classifier_regression_targets, axis = 0)
    std_ty, std_tx, std_th, std_tw = np.std(self._classifier_regression_targets, axis = 0)
    print("Classifier Regression Target Means : %1.2f %1.2f %1.2f %1.2f" % (mean_ty, mean_tx, mean_th, mean_tw))
    print("Classifier Regression Target StdDev: %1.2f %1.2f %1.2f %1.2f" % (std_ty, std_tx, std_th, std_tw))
    mean_ty, mean_tx, mean_th, mean_tw = np.mean(self._classifier_regression_predictions, axis = 0)
    std_ty, std_tx, std_th, std_tw = np.std(self._classifier_regression_predictions, axis = 0)
    print("Classifier Regression Prediction Means : %1.2f %1.2f %1.2f %1.2f" % (mean_ty, mean_tx, mean_th, mean_tw))
    print("Classifier Regression Prediction StdDev: %1.2f %1.2f %1.2f %1.2f" % (std_ty, std_tx, std_th, std_tw))
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
    
    # Compute class accuracy and recall. Note that invalid anchor locations
    # have their corresponding objectness class score set to 0 (neutral). It is
    # therefore safe to determine the total number of positive and negative
    # anchors by inspecting the class score.
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

    # Extract all ground truth regression targets for RPN
    for i in range(y_true.shape[0]):
      for y in range(y_true.shape[1]):
        for x in range(y_true.shape[2]):
          for k in range(y_true.shape[3]):
            if y_true[i,y,x,k,2] > 0:
              targets = y_true[i,y,x,k,4:8]
              self._rpn_regression_targets = np.vstack([self._rpn_regression_targets, targets])  


  def on_classifier_step(self, losses, y_predicted_class, y_predicted_regression, y_true_classes, y_true_regressions):
    i = self._step_number
    self._classifier_total_losses[i] = losses[0]
    self._classifier_class_losses[i] = losses[1]
    self._classifier_regression_losses[i] = losses[2]

    self.classifier_mean_class_loss = np.mean(self._classifier_class_losses[0:i+1])
    self.classifier_mean_regression_loss = np.mean(self._classifier_regression_losses[0:i+1])
    self.classifier_mean_total_loss = self.classifier_mean_class_loss + self.classifier_mean_regression_loss

    # Extract all ground truth regression targets: ty, tx, th, tw
    assert len(y_true_regressions.shape) == 4 and y_true_regressions.shape[0] == 1  # only batch size of 1 currently supported
    for n in range(y_true_regressions.shape[1]):
      indices = np.nonzero(y_true_regressions[0,n,0,:])[0]  # valid mask
      assert indices.size == 4 or indices.size == 0
      if indices.size == 4:
        targets = y_true_regressions[0,n,1][indices]        # ty, tx, th, tw
        self._classifier_regression_targets = np.vstack([self._classifier_regression_targets, targets])
    # Do the same for predictions
    assert len(y_predicted_regression.shape) == 3 and y_predicted_regression.shape[0] == 1
    assert len(y_predicted_class.shape) == 3 and y_predicted_class.shape[0] == 1
    for n in range(y_predicted_regression.shape[1]):
      class_idx = np.argmax(y_predicted_class[0,n])
      if class_idx > 0:
        idx = class_idx - 1
        predictions = y_predicted_regression[0,n,idx*4:idx*4+4]
        self._classifier_regression_predictions = np.vstack([self._classifier_regression_predictions, predictions])       
      
  def on_step_end(self):
    """
    Must be called at the end of each training step after all the other step functions.
    """
    self._step_number += 1

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

   
def dump(proposals, ground_truth_object_boxes, y_true_proposal_classes, y_true_proposal_regressions):
  print("--")
  for box in ground_truth_object_boxes:
    print("Box: (%d) %d %d %d %d" % (box.class_index, box.y_min, box.x_min, box.y_max, box.x_max))
  for i in range(proposals.shape[0]):
    class_idx = np.argmax(y_true_proposal_classes[i])
    if class_idx > 0:
      idx = class_idx - 1
      ty, tx, th, tw = y_true_proposal_regressions[i,1,idx*4:idx*4+4]
      proposal_width = proposals[i,3] - proposals[i,1] + 1
      proposal_height = proposals[i,2] - proposals[i,0] + 1
      proposal_center_x = 0.5 * (proposals[i,1] + proposals[i,3])
      proposal_center_y = 0.5 * (proposals[i,0] + proposals[i,2])
      center_y = ty * proposal_height + proposal_center_y
      center_x = tx * proposal_width + proposal_center_x
      width = exp(tw) * proposal_width
      height = exp(th) * proposal_height
      y1, x1, y2, x2 = (center_y - 0.5 * height, center_x - 0.5 * width, center_y + 0.5 * height, center_x + 0.5 * width)
      print("Proposal %d %d %d %d -> (%d) %d %d %d %d" % (proposals[i,0], proposals[i,1], proposals[i,2], proposals[i,3], class_idx, y1, x1, y2, x2)) 
    else:
      print("Proposal %d %d %d %d -> (%d)" % (proposals[i,0], proposals[i,1], proposals[i,2], proposals[i,3], 0)) 

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
  y_true_proposal_classes, y_true_proposal_regressions = region_proposal_network.label_proposals(proposals = proposals, ground_truth_object_boxes = ground_truth_object_boxes, num_classes = num_classes)
  #dump(proposals, ground_truth_object_boxes, y_true_proposal_classes, y_true_proposal_regressions)

  # Sample from proposals
  proposals, y_true_proposal_classes, y_true_proposal_regressions = sample_proposals(proposals = proposals, y_true_proposal_classes = y_true_proposal_classes, y_true_proposal_regressions = y_true_proposal_regressions, max_proposals = 4, positive_fraction = 0.5)
  
  # Convert to anchor map (RPN output map) space
  proposals = vgg16.convert_box_coordinates_from_image_to_output_map_space(box = proposals, output_map_shape = cnn_output_shape)

  # Convert from (y_min,x_min,y_max,x_max) -> (y_min,x_min,height,width) as expected by RoI pool layer
  proposals[:,2:4] = proposals[:,2:4] - proposals[:,0:2] + 1

  # RoI pooling layer expects tf.int32
  proposals = proposals.astype(np.int32)

  # Reshape to batch size of 1 (e.g., (N,M) -> (1,N,M))
  proposals = np.expand_dims(proposals, axis = 0)
  y_true_proposal_classes = np.expand_dims(y_true_proposal_classes, axis = 0)
  y_true_proposal_regressions = np.expand_dims(y_true_proposal_regressions, axis = 0)

  return proposals, y_true_proposal_classes, y_true_proposal_regressions

# good test images:
# 2010_004041.jpg
# 2010_005080.jpg
# 2008_000019.jpg
# 2009_004872.jpg
if __name__ == "__main__":
  """
  y_true = np.array([
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
  ]).reshape((2,2,4))
  y_predicted = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 1]
  ]).reshape((2,2,4))

  loss = K.eval(classifier_class_loss(y_true = K.variable(y_true), y_predicted = K.variable(y_predicted)))
  print(loss)
  print("---")
  for j in range(2):
    for i in range(2):
      yt = K.variable(y_true[j,i])
      yp = K.variable(y_predicted[j,i])
      print(K.eval(K.categorical_crossentropy(yt, yp)))

  exit()
  
  y_true = np.zeros((1,2,2,8))
  y_true[0,0,0,:] = 1, 1, 1, 1, 0, 0, 0, 0
  y_true[0,0,1,:] = 1, 2, 3, 3, 2, 2, 2, 2
  y_true[0,1,0,:] = 0, 0, 0, 0, 0, 0, 0, 0
  y_true[0,1,1,:] = 0, 3, 0, 0, 3, 2, 1, 0
  y_predicted = np.array([
    [1,2,3,4,5,6,7,8],
    [0,1,2,3,3,2,1,0]
  ]).reshape((1,2,8))

  loss = K.eval(classifier_regression_loss(y_true = K.variable(y_true), y_predicted = K.variable(y_predicted)))
  print(loss)
  exit()
  """
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
  parser.add_argument("--rpn-only", action = "store_true", help = "Train only the region proposal network")
  parser.add_argument("--save-to", metavar = "filepath", type = str, action = "store", help = "File to save model weights to when training is complete")
  parser.add_argument("--load-from", metavar="filepath", type = str, action = "store", help = "File to load initial model weights from")
  parser.add_argument("--infer-boxes", metavar = "file", type = str, action = "store", help = "Run inference on image using region proposal network and display bounding boxes")
  parser.add_argument("--show-objects", metavar = "file", type = str, action = "store", help = "Run inference on image using classifier network and display bounding boxes")
  options = parser.parse_args()

  voc = VOC(dataset_dir = options.dataset_dir, scale = 600)

  rpn_model, conv_model = build_rpn_model(weights_filepath = options.load_from, learning_rate = options.learning_rate, clipnorm = options.clipnorm, l2 = options.l2)
  classifier_model = build_classifier_model(num_classes = voc.num_classes, conv_model = conv_model, weights_filepath = options.load_from, learning_rate = options.learning_rate, clipnorm = options.clipnorm)
  complete_model = build_complete_model(rpn_model = rpn_model, classifier_model = classifier_model) # contains all weights, used for saving weights
  complete_model.summary()

  if options.show_image:
    show_image(voc = voc, filename = options.show_image)

  if options.infer_boxes:
    infer_rpn_boxes(rpn_model = rpn_model, voc = voc, filename = options.infer_boxes)

  if options.show_objects:
    show_objects(rpn_model = rpn_model, classifier_model = classifier_model, voc = voc, filename = options.show_objects)

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

        # TODO: Better separation of RPN and classifier

        # Fetch one sample and reshape to batch size of 1
        # TODO: should we just return complete y_true with a y_batch/y_valid map to define mini-batch?
        image_path, x, y_true_minibatch, anchor_boxes = next(train_data)
        input_image_shape = x.shape
        cnn_output_shape = vgg16.compute_output_map_shape(input_image_shape = input_image_shape)
        image_info = voc.get_image_description(image_path)
        ground_truth_object_boxes = image_info.get_boxes()              #TODO: return this from iterator so we don't need image_info
        y_true = image_info.get_complete_ground_truth_regressions_map() #TODO: ""
        y_true = np.expand_dims(y_true, axis = 0)
        y_true_minibatch = np.expand_dims(y_true_minibatch, axis = 0)
        x = np.expand_dims(x, axis = 0)

        # RPN: back prop one step (and then predict so we can evaluate accuracy)
        rpn_losses = rpn_model.train_on_batch(x = x, y = y_true_minibatch) # loss = [sum, loss_cls, loss_regr]
        y_predicted_class, y_predicted_regression = rpn_model.predict_on_batch(x = x)

        # Classifier
        if not options.rpn_only:
          proposals = region_proposal_network.extract_proposals(y_predicted_class = y_predicted_class, y_predicted_regression = y_predicted_regression, y_true = y_true, anchor_boxes = anchor_boxes)
          if proposals.shape[0] > 0:
            # Prepare proposals for input to classifier network and generate
            # labels
            proposals, y_true_proposal_classes, y_true_proposal_regressions = convert_proposals_to_classifier_network_format(
              proposals = proposals,
              input_image_shape = input_image_shape,
              cnn_output_shape = cnn_output_shape,
              ground_truth_object_boxes = ground_truth_object_boxes,
              num_classes = voc.num_classes)

            # Do we have any proposals to process?
            if proposals.size > 0:
              # Classifier: back prop one step (and then predict)
              classifier_losses = classifier_model.train_on_batch(x = [ x, proposals ], y = [ y_true_proposal_classes, y_true_proposal_regressions ])
              y_classifier_predicted_class, y_classifier_predicted_regression = classifier_model.predict_on_batch(x = [ x, proposals ])

              # Update classifier progress
              stats.on_classifier_step(losses = classifier_losses, y_predicted_class = y_classifier_predicted_class, y_predicted_regression = y_classifier_predicted_regression, y_true_classes = y_true_proposal_classes, y_true_regressions = y_true_proposal_regressions)

        # Update RPN progress and progress bar
        stats.on_rpn_step(losses = rpn_losses, y_predicted_class = y_predicted_class, y_predicted_regression = y_predicted_regression, y_true_minibatch = y_true_minibatch, y_true = y_true)
        progbar.update(current = i, values = [ 
          ("rpn_total_loss", stats.rpn_mean_total_loss),
          ("rpn_class_loss", stats.rpn_mean_class_loss),
          ("rpn_regression_loss", stats.rpn_mean_regression_loss),
          ("rpn_class_accuracy", stats.rpn_mean_class_accuracy),
          ("rpn_class_recall", stats.rpn_mean_class_recall),
          ("classifier_total_loss", stats.classifier_mean_total_loss),
          ("classifier_class_loss", stats.classifier_mean_class_loss),
          ("classifier_regression_loss", stats.classifier_mean_regression_loss)
        ])
        stats.on_step_end()

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
