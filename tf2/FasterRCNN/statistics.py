#
# Faster R-CNN in PyTorch and TensorFlow 2 w/ Keras
# tf2/FasterRCNN/statistics.py
# Copyright 2021-2022 Bart Trzynadlowski
#
# Classes for computing training-related statistics.
#

from collections import defaultdict
import numpy as np

from .models.math_utils import intersection_over_union


class TrainingStatistics:
  """
  Computes statistics per epoch.
  """
  def __init__(self):
    self.rpn_class_loss = float("inf")
    self.rpn_regression_loss = float("inf")
    self.detector_class_loss = float("inf")
    self.detector_regression_loss = float("inf")
    self._rpn_class_losses = []
    self._rpn_regression_losses = []
    self._detector_class_losses = []
    self._detector_regression_losses = []

  def on_training_step(self, losses):
    """
    Call once per training iteration to aggregate losses.

    Parameters
    ----------
    losses : models.faster_rcnn.FasterRCNNModel.Loss
      Dataclass containing losses broken down by RPN and detector, and further
      by classifier and regression loss. 
    """
    self._rpn_class_losses.append(losses["rpn_class_loss"])
    self._rpn_regression_losses.append(losses["rpn_regression_loss"])
    self._detector_class_losses.append(losses["detector_class_loss"])
    self._detector_regression_losses.append(losses["detector_regression_loss"])
    self.rpn_class_loss = np.mean(self._rpn_class_losses)
    self.rpn_regression_loss = np.mean(self._rpn_regression_losses)
    self.detector_class_loss = np.mean(self._detector_class_losses)
    self.detector_regression_loss = np.mean(self._detector_regression_losses)

  def get_progbar_postfix(self):
    """
    Returns
    -------
    Dict[str, str]
      A dictionary of labels and values suitable for use as a postfix object
      for a tqdm progress bar.
    """
    return { 
      "rpn_class_loss": "%1.4f" % self.rpn_class_loss,
      "rpn_regr_loss": "%1.4f" % self.rpn_regression_loss,
      "detector_class_loss": "%1.4f" % self.detector_class_loss,
      "detector_regr_loss": "%1.4f" % self.detector_regression_loss,
      "total_loss": "%1.2f" % (self.rpn_class_loss + self.rpn_regression_loss + self.detector_class_loss + self.detector_regression_loss)
    }


class PrecisionRecallCurveCalculator:
  """
  Collects data over the course of a validation pass and then computes
  precision and recall (including mean average precision).
  """
  def __init__(self):
    # List of (confidence_score, correctness) by class for all images in dataset
    self._unsorted_predictions_by_class_index = defaultdict(list)

    # True number of objects by class for all images in dataset
    self._object_count_by_class_index = defaultdict(int)

  def _compute_correctness_of_predictions(self, scored_boxes_by_class_index, gt_boxes):
    unsorted_predictions_by_class_index = {}
    object_count_by_class_index = defaultdict(int)

    # Count objects by class. We do this here because in case there are no
    # predictions, we do not want to miscount the total number of objects.
    for gt_box in gt_boxes:
      object_count_by_class_index[gt_box.class_index] += 1

    for class_index, scored_boxes in scored_boxes_by_class_index.items():
      # Get the ground truth boxes corresponding to this class
      gt_boxes_this_class = [ gt_box for gt_box in gt_boxes if gt_box.class_index == class_index ]

      # Compute IoU of each box with each ground truth box and store as a list
      # of tuples (iou, box_index, gt_box_index) by descending IoU
      ious = []
      for gt_idx in range(len(gt_boxes_this_class)):
        for box_idx in range(len(scored_boxes)):
          boxes1 = np.expand_dims(scored_boxes[box_idx][0:4], axis = 0) # convert single box (4,) to (1,4), as expected by parallel IoU function
          boxes2 = np.expand_dims(gt_boxes_this_class[gt_idx].corners, axis = 0)
          iou = intersection_over_union(boxes1 = boxes1, boxes2 = boxes2) 
          ious.append((iou, box_idx, gt_idx))
      ious = sorted(ious, key = lambda iou: ious[0], reverse = True)  # sort descending by IoU
      
      # Vector that indicates whether a ground truth box has been detected
      gt_box_detected = [ False ] * len(gt_boxes)

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
        if is_true_positive[box_idx] or gt_box_detected[gt_idx]:
          # The prediction and/or ground truth box have already been matched
          continue
        # We've got a true positive
        is_true_positive[box_idx] = True
        gt_box_detected[gt_idx] = True
      # Construct the final array of prediction descriptions
      unsorted_predictions_by_class_index[class_index] = [ (scored_boxes[i][4], is_true_positive[i]) for i in range(len(scored_boxes)) ]
        
    return unsorted_predictions_by_class_index, object_count_by_class_index

  def add_image_results(self, scored_boxes_by_class_index, gt_boxes):
    """
    Adds a detection result to the running tally. Should be called only once per
    image in the dataset.

    Parameters
    ----------
    scored_boxes_by_class_index : Dict[int, Tuple[float, float, float, float, float]]
      Final detected boxes as lists of tuples, (y_min, x_min, y_max, x_max,
      score), by class index. The score is the softmax output and is
      interpreted as a confidence metric when sorting results for the mAP
      calculation.
    gt_boxes : List[datasets.training_sample.Box]
      A list of datasets.training_sample.Box objects describing all ground
      truth boxes in the image.
    """
    # Merge in results for this single image
    unsorted_predictions_by_class_index, object_count_by_class_index = self._compute_correctness_of_predictions(
      scored_boxes_by_class_index = scored_boxes_by_class_index,
      gt_boxes = gt_boxes) 
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

    # Insert 0 at the beginning and end of the list. The 0 at the beginning won't
    # matter due to how interpolation works, below.
    recall_array.insert(0, 0.0)
    recall_array.append(1.0)
    precision_array.insert(0, 0.0)
    precision_array.append(0.0)

    # Interpolation means we compute the highest precision observed at a given
    # recall value. Specifically, it means taking the maximum value seen from
    # each point onward. See URL below:
    # https://towardsdatascience.com/breaking-down-mean-average-precision-map-ae462f623a52#1a59
    for i in range(len(precision_array)):
      precision_array[i] = np.max(precision_array[i:])
    
    # Compute AP using simple rectangular integration under the curve
    average_precision = 0
    for i in range(len(recall_array) - 1):
      dx = recall_array[i + 1] - recall_array[i + 0]
      dy = precision_array[i + 1]
      average_precision += dy * dx

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
  
  def plot_precision_vs_recall(self, class_index, class_name = None, interpolated = False):
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
    average_precision, recall_array, precision_array = self._compute_average_precision(class_index = class_index, interpolated = interpolated)

    # Plot raw precision vs. recall
    import matplotlib.pyplot as plt
    label = "{0} AP={1:1.2f}".format("Class {}".format(class_index) if class_name is None else class_name, average_precision)
    plt.plot(recall_array, precision_array, label = label)
    if interpolated:
      plt.title("Precision (Interpolated) vs. Recall")
    else:
      plt.title("Precision vs. Recall")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.show()
    plt.clf()

  def plot_average_precisions(self, class_index_to_name): 
    # Compute average precisions for each class
    labels = [ class_index_to_name[class_index] for class_index in self._object_count_by_class_index ]
    average_precisions = []
    for class_index in self._object_count_by_class_index:
      average_precision, _, _ = self._compute_average_precision(class_index = class_index)
      average_precisions.append(average_precision)

    # Sort alphabetically by class name
    sorted_results = sorted(zip(labels, average_precisions), reverse = True, key = lambda pair: pair[0])
    labels, average_precisions = zip(*sorted_results) # unzip 
    
    # Convert to %
    average_precisions = np.array(average_precisions) * 100.0 # convert to %

    # Bar plot
    import matplotlib.pyplot as plt
    plt.clf()
    plt.xlim([0, 100])
    plt.barh(labels, average_precisions)
    plt.title("Model Performance")
    plt.xlabel("Average Precision (%)")
    for index, value in enumerate(average_precisions):
      plt.text(value, index, "%1.1f" % value)
    plt.show()

  def print_average_precisions(self, class_index_to_name):
    # Compute average precisions for each class
    labels = [ class_index_to_name[class_index] for class_index in self._object_count_by_class_index ]
    average_precisions = []
    for class_index in self._object_count_by_class_index:
      average_precision, _, _ = self._compute_average_precision(class_index = class_index)
      average_precisions.append(average_precision)

    # Sort by score (descending)
    sorted_results = sorted(zip(labels, average_precisions), reverse = True, key = lambda pair: pair[1])
    _, average_precisions = zip(*sorted_results) # unzip

    # Maximum width of any class name (for pretty printing)
    label_width = max([ len(label) for label in labels ])

    # Pretty print
    print("Average Precisions")
    print("------------------")
    for (label, average_precision) in sorted_results:
      print("%s: %1.1f%%" % (label.ljust(label_width), average_precision * 100.0))
    print("------------------")
