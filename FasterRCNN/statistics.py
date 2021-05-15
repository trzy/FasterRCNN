#
# FasterRCNN for Keras
# Copyright 2021 Bart Trzynadlowski
#
# statistics.py
#
# Statistics calculations for assessing training and validation performance.
#

from collections import defaultdict
import numpy as np


class Statistics:
  def __init__(self, num_samples):
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

    self._timing_samples = defaultdict(list)

  def _update_timings(self, timing_samples):
    for label, sample in timing_samples.items():
      self._timing_samples[label].append(sample)

  def on_epoch_begin(self):
    """
    Must be called at the beginning of each epoch.
    """
    self._step_number = 0
    self._rpn_regression_targets = np.zeros((0,4))
    self._classifier_regression_targets = np.zeros((0,4))
    self._classifier_regression_predictions = np.zeros((0,4))
    self._timing_samples.clear()

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
    # Print profiling statistics
    max_times = { label: max(samples) for label, samples in self._timing_samples.items() }
    mean_times = { label: np.average(samples) for label, samples in self._timing_samples.items() }
    print("Max times : %s" % str(max_times))
    print("Mean times: %s" % str(mean_times))

  def on_step_begin(self):
    """
    Must be called at the beginning of each training step before the other step
    update functions (e.g., on_rpn_step()).
    """
    pass

  def on_rpn_step(self, losses, y_predicted_class, y_predicted_regression, y_true_minibatch, y_true, timing_samples):
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
        region_proposal_network.compute_ground_truth_map() for layout.
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

    # Update profiling stats
    self._update_timings(timing_samples = timing_samples)

  def on_classifier_step(self, losses, y_predicted_class, y_predicted_regression, y_true_classes, y_true_regressions, timing_samples):
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
      
    # Update profiling stats
    self._update_timings(timing_samples = timing_samples)
      
  def on_step_end(self):
    """
    Must be called at the end of each training step after all the other step functions.
    """
    self._step_number += 1


