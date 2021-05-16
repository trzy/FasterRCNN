#
# FasterRCNN for Keras
# Copyright 2021 Bart Trzynadlowski
#
# statistics.py
#
# Statistics calculations for assessing training and validation performance.
#

from .models.intersection_over_union import intersection_over_union

from collections import defaultdict
import numpy as np


class AveragePrecision:
  """
  Collects data over the course of a validation pass and then computes mean
  average precision.
  """
  def __init__(self):
    # List of (confidence_score, correctness) by class for all images in dataset
    self._unsorted_predictions_by_class_index = defaultdict(list)

    # True number of objects by class for all images in dataset
    self._object_count_by_class_index = defaultdict(int)

  def _compute_correctness_of_predictions(self, scored_boxes_by_class_index, ground_truth_object_boxes):
    unsorted_predictions_by_class_index = {}
    object_count_by_class_index = defaultdict(int)

    # Count objects by class. We do this here because in case there are no
    # predictions, we do not want to miscount the total number of objects.
    for ground_truth_box in ground_truth_object_boxes:
      object_count_by_class_index[ground_truth_box.class_index] += 1

    for class_index, scored_boxes in scored_boxes_by_class_index.items():
      # Get the ground truth boxes corresponding to this class
      ground_truth_boxes_this_class = [ ground_truth_box for ground_truth_box in ground_truth_object_boxes if ground_truth_box.class_index == class_index ]

      # Compute IoU of each box with each ground truth box and store as a list
      # of tuples (iou, box_index, ground_truth_box_index) by descending IoU
      ious = []
      for gt_idx in range(len(ground_truth_boxes_this_class)):
        for box_idx in range(len(scored_boxes)):
          iou = intersection_over_union(box1 = scored_boxes[box_idx][0:4], box2 = ground_truth_boxes_this_class[gt_idx].corners)
          ious.append((iou, box_idx, gt_idx))
      ious = sorted(ious, key = lambda iou: ious[0], reverse = True)  # sort descending by IoU
      
      # Vector that indicates whether a ground truth box has been detected
      ground_truth_box_detected = [ False ] * len(ground_truth_object_boxes)

      # Vector that indicates whether a prediction is a true positive (True) or
      # false positive (False)
      is_true_positive = [ False ] * len(scored_boxes)
      
      #
      # Construct a list of prediction descriptions: (score, correct)
      # Score is the confidence score of the predicted box and correct is
      # whether it is a true positive (True) or false positive (False).
      #
      # A true positive is a prediction that has an IoU of > 0.5 and is
      # also the highest-IoU prediction for a ground truth box. Predictions
      # with IoU <= 0.5 or that do not have the highest IoU for any ground
      # truth box are considered false positives.
      #
      iou_threshold = 0.5
      for iou, box_idx, gt_idx in ious:
        if iou <= iou_threshold:
          continue
        if is_true_positive[box_idx] or ground_truth_box_detected[gt_idx]:
          # The prediction and/or ground truth box have already been matched
          continue
        # We've got a true positive
        is_true_positive[box_idx] = True
        ground_truth_box_detected[gt_idx] = True
      # Construct the final array of prediction descriptions
      unsorted_predictions_by_class_index[class_index] = [ (scored_boxes[i][4], is_true_positive[i]) for i in range(len(scored_boxes)) ]
        
    return unsorted_predictions_by_class_index, object_count_by_class_index

  def add_image_results(self, scored_boxes_by_class_index, ground_truth_object_boxes):
    """
    Adds a detection result to the running tally. Should be called only once per
    image in the dataset.

    Parameters
    ----------
      scored_boxes_by_class_index : dict
        Final detected boxes as lists of tuples, (y_min, x_min, y_max, x_max,
        score), by class index. The score is the softmax output and is
        interpreted as a confidence metric when sorting results for the mAP
        calculation.
      ground_truth_object_boxes : list
        A list of VOC.Box objects describing all ground truth boxes in the
        image.
    """
    # Merge in results for this single image
    unsorted_predictions_by_class_index, object_count_by_class_index = self._compute_correctness_of_predictions(
      scored_boxes_by_class_index = scored_boxes_by_class_index,
      ground_truth_object_boxes = ground_truth_object_boxes) 
    for class_index, predictions in unsorted_predictions_by_class_index.items():
      self._unsorted_predictions_by_class_index[class_index] += predictions
    for class_index, count in object_count_by_class_index.items():
      self._object_count_by_class_index[class_index] += object_count_by_class_index[class_index]

  def _compute_average_precision(self, class_index):
    # Sort predictions in descending order of score
    sorted_predictions = sorted(self._unsorted_predictions_by_class_index[class_index], key = lambda prediction: prediction[0], reverse = True)
    num_ground_truth_positives = self._object_count_by_class_index[class_index]

    # Compute raw recall and precision arrays
    recall_array = []
    precision_array = []
    true_positives = 0  # running tally
    false_positives = 0 # ""
    for i in range(len(sorted_predictions)):
      true_positives += 1 if sorted_predictions[i][1] == True else 0
      false_positives += 0 if sorted_predictions[i][1] == True else 1
      recall = true_positives / num_ground_truth_positives
      precision = true_positives / (true_positives + false_positives)
      recall_array.append(recall)
      precision_array.append(precision)

    # Compute AP by integrating under the curve. We do not interpolate
    # precision value to remove increases (as is done in the URL below). Numpy
    # seems perfectly capable of handling x arrays with repeating values.
    # https://towardsdatascience.com/breaking-down-mean-average-precision-map-ae462f623a52#1a59
    average_precision = np.trapz(x = recall_array, y = precision_array)

    return average_precision, recall_array, precision_array

  def compute_mean_average_precision(self):
    """
    Calculates mAP (mean average precision) using all the data accumulated thus
    far. This should be called only after all image results have been
    processed.

    Returns
    -------
    np.float64
      Mean average precision.
    """
    average_precisions = []
    for class_index in self._object_count_by_class_index:
      average_precision, _, _ = self._compute_average_precision(class_index = class_index)
      average_precisions.append(average_precision)
    return np.mean(average_precisions)
  
  def plot_precision_vs_recall(self, class_index, class_name = None):
    """
    Plots precision (y axis) vs. recall (x axis) using all the data accumulated
    thus far. This should be called only after all image results have been
    processed.

    Parameters
    ----------
      class_index : int
        The class index for which the curve is plotted.
      class_name : str
        If given, used as the class name on the plot label. Otherwise, the
        numeric class index is used directly.
    """
    average_precision, recall_array, precision_array = self._compute_average_precision(class_index = class_index)

    # Plot raw precision vs. recall
    import matplotlib.pyplot as plt
    label = "{0} AP={1:1.2f}".format("Class {}".format(class_index) if class_name is None else class_name, average_precision)
    plt.plot(recall_array, precision_array, label = label)
    plt.title("Precision vs. Recall")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.show()
    plt.clf()


class ModelStatistics:
  """
  Maintains statistics during training or validation on model performance,
  namely loss and accuracy as well as some performance profiling. Many of the
  tracked quantities are designed to be printed on the Keras progress bar.
  """
  def __init__(self, num_samples):
    """
    Constructor.

    Parameters
    ----------
      num_samples : int
        Number of samples per epoch, used to pre-allocate arrays.
    """
    self._num_samples = num_samples
    self._reset()

  def _reset(self):
    num_samples = self._num_samples

    # The classifier model is not run during steps where no proposals are
    # generated, hence we need different step counts
    self._rpn_step_number = 0
    self._classifier_step_number = 0

    self._rpn_total_losses = np.zeros(num_samples)
    self._rpn_class_losses = np.zeros(num_samples)
    self._rpn_regression_losses = np.zeros(num_samples)
    self._rpn_num_ground_truth_positives = 0
    self._rpn_num_ground_truth_negatives = 0
    self._rpn_num_true_positives = 0
    self._rpn_num_true_negatives = 0
    self._rpn_num_total_samples = 0

    self._classifier_total_losses = np.zeros(num_samples)
    self._classifier_class_losses = np.zeros(num_samples)
    self._classifier_regression_losses = np.zeros(num_samples)

    self._rpn_regression_targets = np.zeros((0,4))
    self._classifier_regression_targets = np.zeros((0,4))
    self._classifier_regression_predictions = np.zeros((0,4))

    self.rpn_mean_class_loss = float("inf")
    self.rpn_class_accuracy = 0
    self.rpn_class_recall = 0
    self.rpn_mean_regression_loss = float("inf")
    self.rpn_mean_total_loss = float("inf")
    
    self.classifier_mean_class_loss = float("inf")
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
    self._reset()

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
    Must be called at the beginning of each step before the other step update
    functions (e.g., on_rpn_step()).
    """
    pass

  def on_rpn_step(self, losses, y_predicted_class, y_predicted_regression, y_true_minibatch, y_true, timing_samples):
    """
    Must be called on each step after the RPN model has been updated. Updates
    the training statistics for the RPN model. Do not call more than once per
    training step. Order relative to the classifier step is irrelevant.

    Parameters
    ----------
      losses : list 
        RPN model losses from Keras train_on_batch() as a 3-element array,
        [ total_loss, class_loss, regression_loss ].
      y_predicted_class : np.ndarray
        RPN model objectness classification output of shape (1, height, width,
        k), where k is the number of anchors. Each element indicates the
        corresponding anchor is an object (>0.5) or background (<0.5).
      y_predicted_regression : np.ndarray
        RPN model regression outputs, with shape (1, height, width, k*4).
      y_true_minibatch : np.ndarray
        RPN ground truth map for the mini-batch used in this training step. The
        map contains ground truth regression targets and object classes and,
        most importantly, a mask indicating which anchors are valid and were
        used in the mini-batch. See
        region_proposal_network.compute_ground_truth_map() for layout.
      y_true : np.ndarray
        Complete RPN ground truth map for all anchors in the image (the anchor
        valid mask indicates all valid anchors from which mini-batches
        are drawn). This is used to compute classification accuracy and recall
        statistics because predictions occur over all possible anchors in the
        image.
      timing_samples : dict
        Performance profiling samples as a dictionary mapping labels to floats.
        Timings are aggregated by label, which is an arbitrary string.
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

    # Update progress
    i = self._rpn_step_number
    self._rpn_total_losses[i] = losses[0]
    self._rpn_class_losses[i] = losses[1]
    self._rpn_regression_losses[i] = losses[2]
    self._rpn_num_ground_truth_positives += num_ground_truth_positives
    self._rpn_num_ground_truth_negatives += num_ground_truth_negatives
    self._rpn_num_true_positives += true_positives
    self._rpn_num_true_negatives += true_negatives
    self._rpn_num_total_samples += total_samples
    
    self.rpn_mean_class_loss = np.mean(self._rpn_class_losses[0:i+1])
    self.rpn_class_accuracy = (self._rpn_num_true_positives + self._rpn_num_true_negatives) / self._rpn_num_total_samples
    self.rpn_class_recall = self._rpn_num_true_positives / self._rpn_num_ground_truth_positives
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
    
    # Increment step count
    self._rpn_step_number += 1

  def on_classifier_step(self, losses, y_predicted_class, y_predicted_regression, y_true_classes, y_true_regressions, timing_samples):
    """
    Must be called each step after the classifier model has been updated but
    only if any objects were detected. Order relative to the RPN step is
    irrelevant.

    Parameters
    ----------
      losses : list 
        Classifier model losses from Keras train_on_batch() as a 3-element
        array, [ total_loss, class_loss, regression_loss ].
      y_predicted_class : np.ndarray
        Class predictions, (1,N,num_classes) tensor, one-hot encoded for each
        detection.
      y_predicted_regression : np.ndarray
        Predicted regression parameters, (1,N,(num_classes-1)*4), four values
        for each non-background class (1 and up): (ty, tx, th, tw).
      y_true_classes : np.ndarray
        Ground truth classes, (1,N,num_classes).
      y_true_regressions : np.ndarray
        Ground truth regression targets, (1,N,2,(num_classes-1)*4). Targets are
        stored as [:,:,1,:] and a mask of which 4 values are valid appears in
        [:,:,0,:].
      timing_samples : dict
        Performance profiling samples as a dictionary mapping labels to floats.
        Timings are aggregated by label, which is an arbitrary string.
    """
    i = self._classifier_step_number
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
    
    # Increment step count
    self._classifier_step_number += 1
      
  def on_step_end(self):
    """
    Must be called at the end of each step after all the other step functions.
    """
    pass
