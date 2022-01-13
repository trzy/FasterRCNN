#
# FasterRCNN in PyTorch and TensorFlow 2 w/ Keras
# python/tf2/FasterRCNN/__main__.py
# Copyright 2021-2022 Bart Trzynadlowski
#
# Main module for the TensorFlow/Keras implementation of FasterRCNN. Run this
# from the root directory, e.g.:
#
# python -m python.tf2.FasterRCNN --help
#


# TODO
# ----
# - Remove redundant stop_gradient in detector and test again. Make a comment about the importance of stop_gradients and a note to investigate further, particularly in README.md
# - Perform test without logits
# - Perform test with old pooling code
# - Remove image_shape_map and just pass image shape when needed
# - FasterRCNN model should be a class with methods to load weights, freeze layers, etc.,
#   as well as prediction code that returns scored boxes
# - Freezing layers should be part of model construction
# - box regressions -> box deltas here and in PyTorch version?
# - Verify mAP using external program
# - Document why L2 = 0.5 * weight decay
# - Move Keras box regression -> box code, and IoU code, from faster_rcnn.py to math_utils as well.
# - In voc.py and here in command line args, and for PyTorch, change default path to dataset to ../VOCdevkit/VOC2007
#
import argparse
import numpy as np
import os
import random
from tqdm import tqdm
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

from .statistics import TrainingStatistics
from .statistics import PrecisionRecallCurveCalculator
from .datasets import voc
from .models import faster_rcnn
from .models import vgg16
from .models import math_utils
from .models import anchors
from . import visualize

class BestWeightsTracker:
  def __init__(self, filepath):
    self._filepath = filepath
    self._best_weights = None
    self._best_mAP = 0

  def on_epoch_end(self, model, mAP):
    if mAP > self._best_mAP:
      self._best_mAP = mAP
      self._best_weights = model.get_weights()

  def restore_and_save_best_weights(self, model):
    if self._best_weights is not None:
      model.set_weights(self._best_weights)
      model.save_weights(filepath = self._filepath, overwrite = True, save_format = "h5")
      print("Saved best model weights (Mean Average Precision = %1.2f%%) to '%s'" % (self._best_mAP, self._filepath))

def render_anchors():
  training_data = voc.Dataset(dir = options.dataset_dir, split = options.train_split, augment = False, shuffle = False)
  if not os.path.exists(options.dump_anchors):
    os.makedirs(options.dump_anchors)
  print("Rendering anchors from '%s' to set '%s'..." % (options.train_split, options.dump_anchors))
  for sample in iter(training_data):
    output_path = os.path.join(options.dump_anchors, "anchors_" + os.path.basename(sample.filepath) + ".png")
    visualize.show_anchors(
      output_path = output_path,
      image = sample.image,
      anchor_map = sample.anchor_map,
      anchor_valid_map = sample.anchor_valid_map,
      gt_rpn_map = sample.gt_rpn_map,
      gt_boxes = sample.gt_boxes
    )

#TODO: make this part of the Model class
def _copy_weights(dest_model, src_model):
  dest_layers = { layer.name: layer for layer in dest_model.layers }
  src_layers = { layer.name: layer for layer in src_model.layers }
  for name, src_layer in src_layers.items():
    if name in dest_layers:
      dest_layers[name].set_weights(src_layer.get_weights())

def _sample_rpn_minibatch(rpn_map, object_indices, background_indices, rpn_minibatch_size):
  """
  Selects anchors for training and produces a copy of the RPN ground truth
  map with only those anchors marked as trainable.

  Parameters
  ----------
  rpn_map : np.ndarray
    RPN ground truth map of shape
    (batch_size, height, width, num_anchors, 6).
  object_indices : List[np.ndarray]
    For each image in the batch, a map of shape (N, 3) of indices (y, x, k)
    of all N object anchors in the RPN ground truth map.
  background_indices : List[np.ndarray]
    For each image in the batch, a map of shape (M, 3) of indices of all M
    background anchors in the RPN ground truth map.

  Returns
  -------
  np.ndarray
    A copy of the RPN ground truth map with index 0 of the last dimension
    recomputed to include only anchors in the minibatch.
  """
  assert rpn_map.shape[0] == 1, "Batch size must be 1"
  assert len(object_indices) == 1, "Batch size must be 1"
  assert len(background_indices) == 1, "Batch size must be 1"
  positive_anchors = object_indices[0]
  negative_anchors = background_indices[0]
  assert len(positive_anchors) + len(negative_anchors) >= rpn_minibatch_size, "Image has insufficient anchors for RPN minibatch size of %d" % rpn_minibatch_size
  assert len(positive_anchors) > 0, "Image does not have any positive anchors"
  assert rpn_minibatch_size % 2 == 0, "RPN minibatch size must be evenly divisible"

  # Sample, producing indices into the index maps
  num_positive_anchors = len(positive_anchors)
  num_negative_anchors = len(negative_anchors)
  num_positive_samples = min(rpn_minibatch_size // 2, num_positive_anchors) # up to half the samples should be positive, if possible
  num_negative_samples = rpn_minibatch_size - num_positive_samples          # the rest should be negative
  positive_anchor_idxs = random.sample(range(num_positive_anchors), num_positive_samples)
  negative_anchor_idxs = random.sample(range(num_negative_anchors), num_negative_samples)
  
  # Construct index expressions into RPN map
  positive_anchors = positive_anchors[positive_anchor_idxs]
  negative_anchors = negative_anchors[negative_anchor_idxs]
  trainable_anchors = np.concatenate([ positive_anchors, negative_anchors ])
  batch_idxs = np.zeros(len(trainable_anchors), dtype = int)
  trainable_idxs = (batch_idxs, trainable_anchors[:,0], trainable_anchors[:,1], trainable_anchors[:,2], 0)

  # Create a copy of the RPN map with samples set as trainable
  rpn_minibatch_map = rpn_map.copy()
  rpn_minibatch_map[:,:,:,:,0] = 0
  rpn_minibatch_map[trainable_idxs] = 1

  return rpn_minibatch_map

def _predictions_to_scored_boxes(image_data, classes, regressions, proposals, score_threshold):
  # Eliminate batch dimension
  image_data = np.squeeze(image_data, axis = 0)
  classes = np.squeeze(classes, axis = 0)
  regressions = np.squeeze(regressions, axis = 0)

  # Convert logits to probability distribution if using logits mode
  if options.detector_logits:
    classes = tf.nn.softmax(classes, axis = 1).numpy()
  
  # Convert proposal boxes -> center point and size
  proposal_anchors = np.empty(proposals.shape)
  proposal_anchors[:,0] = 0.5 * (proposals[:,0] + proposals[:,2]) # center_y
  proposal_anchors[:,1] = 0.5 * (proposals[:,1] + proposals[:,3]) # center_x
  proposal_anchors[:,2:4] = proposals[:,2:4] - proposals[:,0:2]   # height, width

  # Separate out results per class: class_idx -> (y1, x1, y2, x2, score)
  boxes_and_scores_by_class_idx = {}
  for class_idx in range(1, classes.shape[1]):  # skip class 0 (background)
    # Get the regression parameters (ty, tx, th, tw) corresponding to this
    # class, for all proposals
    regression_idx = (class_idx - 1) * 4
    regression_params = regressions[:, (regression_idx + 0) : (regression_idx + 4)] # (N, 4)
    proposal_boxes_this_class = math_utils.convert_regressions_to_boxes(
      regressions = regression_params,
      anchors = proposal_anchors,
      regression_means = [0.0, 0.0, 0.0, 0.0],
      regression_stds = [0.1, 0.1, 0.2, 0.2]
    )

    # Clip to image boundaries
    proposal_boxes_this_class[:,0::2] = np.clip(proposal_boxes_this_class[:,0::2], 0, image_data.shape[0] - 1)  # clip y1 and y2 to [0,height)
    proposal_boxes_this_class[:,1::2] = np.clip(proposal_boxes_this_class[:,1::2], 0, image_data.shape[1] - 1)  # clip x1 and x2 to [0,width)

    # Get the scores for this class. The class scores are returned in
    # normalized categorical form. Each row corresponds to a class.
    scores_this_class = classes[:,class_idx]

    # Keep only those scoring high enough
    sufficiently_scoring_idxs = np.where(scores_this_class > score_threshold)[0]
    proposal_boxes_this_class = proposal_boxes_this_class[sufficiently_scoring_idxs]
    scores_this_class = scores_this_class[sufficiently_scoring_idxs]
    boxes_and_scores_by_class_idx[class_idx] = (proposal_boxes_this_class, scores_this_class)

  # Perform NMS per class
  scored_boxes_by_class_idx = {}
  for class_idx, (boxes, scores) in boxes_and_scores_by_class_idx.items():
    idxs = tf.image.non_max_suppression(
      boxes = boxes,
      scores = scores,
      max_output_size = proposals.shape[0],
      iou_threshold = 0.3
    )
    idxs = idxs.numpy()
    boxes = boxes[idxs]
    scores = np.expand_dims(scores[idxs], axis = 0) # (N,) -> (N,1)
    scored_boxes = np.hstack([ boxes, scores.T ])   # (N,5), with each row: (y1, x1, y2, x2, score)
    scored_boxes_by_class_idx[class_idx] = scored_boxes

  return scored_boxes_by_class_idx

def _convert_training_sample_to_model_input(sample, mode):
    """
    Converts a training sample obtained from the dataset into an input vector
    that can be passed to the model.

    Parameters
    ----------
    sample : datasets.training_sample.TrainingSample
      Training sample obtained from dataset.
    mode : str
      "train" if the input vector will be fed into a training model otherwise
      "infer".

    Returns
    -------
    List[np.ndarray], np.ndarray, np.ndarray
      Input vector for model (see relevant model definition for details), image
      data, and ground truth RPN minibatch map.. All maps are converted to a
      batch size of 1 as expected by Keras model.
    """

    # Ground truth boxes to NumPy arrays
    gt_box_corners = np.array([ box.corners for box in sample.gt_boxes ]).astype(np.float32)        # (num_boxes,4), where each row is (y1,x1,y2,x2)
    gt_box_class_idxs = np.array([ box.class_index for box in sample.gt_boxes ]).astype(np.int32)   # (num_boxes,), where each is an index [1,num_classes)

    # Expand all maps to a batch size of 1
    image_data = np.expand_dims(sample.image_data, axis = 0)
    image_shape_map = np.array([ [ image_data.shape[1], image_data.shape[2], image_data.shape[3] ] ]) # (1,3), with (height,width,channels)
    anchor_map = np.expand_dims(sample.anchor_map, axis = 0)
    anchor_valid_map = np.expand_dims(sample.anchor_valid_map, axis = 0)
    gt_rpn_map = np.expand_dims(sample.gt_rpn_map, axis = 0)
    gt_rpn_object_indices = [ sample.gt_rpn_object_indices ]
    gt_rpn_background_indices = [ sample.gt_rpn_background_indices ]
    gt_box_corners = np.expand_dims(gt_box_corners, axis = 0)
    gt_box_class_idxs = np.expand_dims(gt_box_class_idxs, axis = 0)

    # Create a RPN minibatch: sample anchors randomly and create a new ground
    # truth RPN map
    gt_rpn_minibatch_map = _sample_rpn_minibatch(
      rpn_map = gt_rpn_map,
      object_indices = gt_rpn_object_indices,
      background_indices = gt_rpn_background_indices,
      rpn_minibatch_size = 256
    )

    # Input vector to model
    if mode == "train":
      x = [ image_data, image_shape_map, anchor_map, anchor_valid_map, gt_rpn_minibatch_map, gt_box_class_idxs, gt_box_corners ]
    else: # "infer"
      x = [ image_data, image_shape_map, anchor_map, anchor_valid_map ]

    # Return all plus some unpacked elements for convenience
    return x, image_data, gt_rpn_minibatch_map

def evaluate(model, eval_data = None, num_samples = None, plot = False, print_average_precisions = False):
  if eval_data is None:
    eval_data = voc.Dataset(dir = options.dataset_dir, split = options.eval_split, augment = False, shuffle = False)
  if num_samples is None:
    num_samples = eval_data.num_samples
  precision_recall_curve = PrecisionRecallCurveCalculator()
  i = 0
  print("Evaluating '%s'..." % eval_data.split)
  for sample in tqdm(iterable = iter(eval_data), total = num_samples):
    x, image_data, _ = _convert_training_sample_to_model_input(sample = sample, mode = "infer")
    _, _, detector_classes, detector_regressions, proposals = model.predict_on_batch(x = x)
    scored_boxes_by_class_index = _predictions_to_scored_boxes(
      image_data = image_data,
      classes = detector_classes,
      regressions = detector_regressions,
      proposals = proposals,
      score_threshold = 0.05
    )
    precision_recall_curve.add_image_results(
      scored_boxes_by_class_index = scored_boxes_by_class_index,
      gt_boxes = sample.gt_boxes
    )
    i += 1
    if i >= num_samples:
      break
  if print_average_precisions:
    precision_recall_curve.print_average_precisions(class_index_to_name = voc.Dataset.class_index_to_name)
  mean_average_precision = 100.0 * precision_recall_curve.compute_mean_average_precision() 
  print("Mean Average Precision = %1.2f%%" % mean_average_precision)
  if plot:
    precision_recall_curve.plot_average_precisions(class_index_to_name = voc.Dataset.class_index_to_name)
  return mean_average_precision

def train(train_model, infer_model):
  print("Training Parameters")
  print("-------------------")
  print("Initial weights           : %s" % (options.load_from if options.load_from else "Keras VGG-16 ImageNet weights"))
  print("Dataset                   : %s" % options.dataset_dir)
  print("Training split            : %s" % options.train_split)
  print("Evaluation split          : %s" % options.eval_split)
  print("Epochs                    : %d" % options.epochs)
  print("Optimizer                 : %s" % options.optimizer)
  print("Learning rate             : %f" % options.learning_rate)
  print("Gradient norm clipping    : %s" % ("disabled" if options.clipnorm <= 0 else ("%f" % options.clipnorm)))
  print("SGD Momentum              : %f" % options.momentum)
  print("Adam Beta-1               : %f" % options.beta1)
  print("Adam Beta-2               : %f" % options.beta2)
  print("Weight decay              : %f" % options.weight_decay)
  print("Dropout                   : %f" % options.dropout)
  print("RoI pooling implementation: %s" % ("custom" if options.custom_roi_pool else "crop-and-resize w/ max pool"))
  print("Detector output           : %s" % ("logits" if options.detector_logits else "probabilities"))
  print("Augmentation              : %s" % ("disabled" if options.no_augment else "enabled"))
  print("Edge proposals            : %s" % ("excluded" if options.exclude_edge_proposals else "included"))
  print("CSV log                   : %s" % ("none" if not options.log_csv else options.log_csv))
  print("Checkpoints               : %s" % ("disabled" if not options.checkpoint_dir else options.checkpoint_dir))
  print("Final weights file        : %s" % ("none" if not options.save_to else options.save_to))
  print("Best weights file         : %s" % ("none" if not options.save_best_to else options.save_best_to))
  training_data = voc.Dataset(dir = options.dataset_dir, split = options.train_split, augment = not options.no_augment, shuffle = True, cache = not options.no_cache)
  eval_data = voc.Dataset(dir = options.dataset_dir, split = options.eval_split, augment = False, shuffle = False, cache = False)
  if options.checkpoint_dir and not os.path.exists(options.checkpoint_dir):
    os.makedirs(options.checkpoint_dir)
  if options.log_csv:
    csv = utils.CSVLog(options.log_csv)
  if options.save_best_to:
    best_weights_tracker = BestWeightsTracker(filepath = options.save_best_to)
  for epoch in range(1, 1 + options.epochs):
    print("Epoch %d/%d" % (epoch, options.epochs))
    stats = TrainingStatistics()
    progbar = tqdm(iterable = iter(training_data), total = training_data.num_samples, postfix = stats.get_progbar_postfix())
    for sample in progbar:
      x, image_data, gt_rpn_minibatch_map = _convert_training_sample_to_model_input(sample = sample, mode = "train")
      losses = train_model.train_on_batch(x = x, y = gt_rpn_minibatch_map, return_dict = True)
      stats.on_training_step(losses = losses)
      progbar.set_postfix(stats.get_progbar_postfix())
    last_epoch = epoch == options.epochs
    _copy_weights(dest_model = infer_model, src_model = train_model)
    mean_average_precision = evaluate(
      model = infer_model,
      eval_data = eval_data,
      num_samples = options.periodic_eval_samples,
      plot = False,
      print_average_precisions = False
    )
    if options.checkpoint_dir:
      checkpoint_file = os.path.join(options.checkpoint_dir, "checkpoint-epoch-%d-mAP-%1.1f.h5" % (epoch, mean_average_precision))
      train_model.save_weights(filepath = checkpoint_file, overwrite = True, save_format = "h5")
      print("Saved model checkpoint to '%s'" % checkpoint_file)
    if options.log_csv:
      log_items = {
        "epoch": epoch,
        "learning_rate": options.learning_rate,
        "clipnorm": options.clipnorm,
        "momentum": options.momentum,
        "beta1": options.beta1,
        "beta2": options.beta2,
        "weight_decay": options.weight_decay,
        "dropout": options.dropout,
        "mAP": mean_average_precision
      }
      log_items.update(stats.get_progbar_postfix())
      csv.log(log_items)
    if options.save_best_to:
      best_weights_tracker.on_epoch_end(model = train_model, mAP = mean_average_precision)
  if options.save_to:
    train_model.save_weights(filepath = options.save_to, overwrite = True, save_format = "h5")
    print("Saved final model weights to '%s'" % options.save_to)
  if options.save_best_to:
    best_weights_tracker.restore_and_save_best_weights(model = train_model)
  print("Evaluating %s model on all samples in '%s'..." % (("best" if options.save_best_to else "final"), options.eval_split))  # evaluate final or best model on full dataset
  evaluate(
    model = infer_model,
    eval_data = eval_data,
    num_samples = eval_data.num_samples,
    plot = options.plot,
    print_average_precisions = True
  )

def _predict(model, image_data, image, show_image, output_path):
  anchor_map, anchor_valid_map = anchors.generate_anchor_maps(image_shape = image_data.shape, feature_pixels = 16)
  anchor_map = np.expand_dims(anchor_map, axis = 0)                                                 # convert to batch size of 1
  anchor_valid_map = np.expand_dims(anchor_valid_map, axis = 0)                                     # ""
  image_data = np.expand_dims(image_data, axis = 0)                                                 # convert to batch size of 1: (1,height,width,3)
  image_shape_map = np.array([ [ image_data.shape[1], image_data.shape[2], image_data.shape[3] ] ]) # (1,3), with (height,width,channels)
  x = [ image_data, image_shape_map, anchor_map, anchor_valid_map ]
  _, _, detector_classes, detector_regressions, proposals = model.predict_on_batch(x = x)
  scored_boxes_by_class_index = _predictions_to_scored_boxes(
    image_data = image_data,
    classes = detector_classes,
    regressions = detector_regressions,
    proposals = proposals,
    score_threshold = 0.7
  )
  visualize.show_detections(
    output_path = output_path,
    show_image = show_image,
    image = image,
    scored_boxes_by_class_index = scored_boxes_by_class_index,
    class_index_to_name = voc.Dataset.class_index_to_name
  )

def predict_one(model, url, show_image, output_path):
  from .datasets.image import load_image
  image_data, image, _, _ = load_image(url = url, min_dimension_pixels = 600)
  _predict(model = model, image_data = image_data, image = image, show_image = show_image, output_path = output_path)

def predict_all(model, split):
  dirname = "predictions_" + split
  if not os.path.exists(dirname):
    os.makedirs(dirname)
  print("Rendering predictions from '%s' set to '%s'..." % (split, dirname))
  dataset = voc.Dataset(dir = options.dataset_dir, split = split, augment = False, shuffle = False)
  for sample in iter(dataset):
    output_path = os.path.join(dirname, os.path.splitext(os.path.basename(sample.filepath))[0] + ".png")
    _predict(model = model, image_data = sample.image_data, image = sample.image, show_image = False, output_path = output_path)

def create_optimizer():
  kwargs = {}
  if options.clipnorm > 0:
    kwargs = { "clipnorm": options.clipnorm }
  if options.optimizer == "sgd":
    optimizer = SGD(learning_rate = options.learning_rate, momentum = options.momentum, **kwargs)
  elif options.optimizer == "adam":
    optimizer = Adam(learning_rate = options.learning_rate, beta_1 = options.beta1, beta_2 = options.beta2, **kwargs)
  else:
    raise ValueError("Optimizer must be 'sgd' for stochastic gradient descent or 'adam' for Adam")
  return optimizer


if __name__ == "__main__":
  parser = argparse.ArgumentParser("FasterRCNN")
  group = parser.add_mutually_exclusive_group()
  group.add_argument("--train", action = "store_true", help = "Train model")
  group.add_argument("--eval", action = "store_true", help = "Evaluate model")
  group.add_argument("--predict", metavar = "url", action = "store", type = str, help = "Run inference on image and display detected boxes")
  group.add_argument("--predict-to-file", metavar = "url", action = "store", type = str, help = "Run inference on image and render detected boxes to 'predictions.png'")
  group.add_argument("--predict-all", metavar = "name", action = "store", type = str, help = "Run inference on all images in the specified dataset split and write to directory 'predictions_${split}'")
  parser.add_argument("--load-from", metavar = "file", action = "store", help = "Load initial model weights from file")
  parser.add_argument("--save-to", metavar = "file", action = "store", help = "Save final trained weights to file")
  parser.add_argument("--save-best-to", metavar = "file", action = "store", help = "Save best weights (highest mean average precision) to file")
  parser.add_argument("--dataset-dir", metavar = "dir", action = "store", default = "../../VOCdevkit/VOC2007", help = "VOC dataset directory")
  parser.add_argument("--train-split", metavar = "name", action = "store", default = "trainval", help = "Dataset split to use for training")
  parser.add_argument("--eval-split", metavar = "name", action = "store", default = "test", help = "Dataset split to use for evaluation")
  parser.add_argument("--no-cache", action = "store_true", help = "Disable image caching during training (reduces memory usage)")
  parser.add_argument("--periodic-eval-samples", metavar = "count", action = "store", default = 1000, help = "Number of samples to use during evaluation after each epoch")
  parser.add_argument("--checkpoint-dir", metavar = "dir", action = "store", help = "Save checkpoints after each epoch to the given directory")
  parser.add_argument("--plot", action = "store_true", help = "Plots the average precision of each class after evaluation (use with --train or --eval)")
  parser.add_argument("--log-csv", metavar = "file", action = "store", help = "Log training metrics to CSV file")
  parser.add_argument("--epochs", metavar = "count", type = int, action = "store", default = 1, help = "Number of epochs to train for")
  parser.add_argument("--optimizer", metavar = "name", type = str, action = "store", default = "sgd", help = "Optimizer to use (\"sgd\" or \"adam\")")
  parser.add_argument("--learning-rate", metavar = "value", type = float, action = "store", default = 1e-3, help = "Learning rate")
  parser.add_argument("--clipnorm", metavar = "value", type = float, action = "store", default = 0.0, help = "Gradient norm clipping (use 0 for none)")
  parser.add_argument("--momentum", metavar = "value", type = float, action = "store", default = 0.9, help = "SGD momentum")
  parser.add_argument("--beta1", metavar = "value", type = float, action = "store", default = 0.9, help = "Adam beta1 parameter (decay rate for 1st moment estimates)")
  parser.add_argument("--beta2", metavar = "value", type = float, action = "store", default = 0.999, help = "Adam beta2 parameter (decay rate for 2nd moment estimates)")
  parser.add_argument("--weight-decay", metavar = "value", type = float, action = "store", default = 0.0, help = "Weight decay")
  parser.add_argument("--dropout", metavar = "probability", type = float, action = "store", default = 0.0, help = "Dropout probability after each of the two fully-connected detector layers")
  parser.add_argument("--custom-roi-pool", action = "store_true", help = "Use custom RoI pool implementation instead of TensorFlow crop-and-resize with max-pool (much slower)")
  parser.add_argument("--detector-logits", action = "store_true", help = "Do not apply softmax to detector class output and compute loss from logits directly")
  parser.add_argument("--no-augment", action = "store_true", help = "Disable image augmentation (random horizontal flips) during training")
  parser.add_argument("--exclude-edge-proposals", action = "store_true", help = "Exclude proposals generated at anchors spanning image edges from being passed to detector stage")
  parser.add_argument("--dump-anchors", metavar = "dir", action = "store", help = "Render out all object anchors and ground truth boxes from the training set to a directory")
  parser.add_argument("--debug-dir", metavar = "dir", action = "store", help = "Enable full TensorFlow Debugger V2 logging to specified directory")
  options = parser.parse_args()

  # Run-time environment
  cuda_available = tf.test.is_built_with_cuda()
  gpu_available = tf.test.is_gpu_available(cuda_only = False, min_cuda_compute_capability = None)
  print("CUDA Available : %s" % ("yes" if cuda_available else "no"))
  print("GPU Available  : %s" % ("yes" if gpu_available else "no"))
  print("Eager Execution: %s" % ("yes" if tf.executing_eagerly() else "no"))

  # Perform optional procedures
  if options.dump_anchors:
    render_anchors()

  # Debug logging
  if options.debug_dir:
    tf.debugging.experimental.enable_dump_debug_info(options.debug_dir, tensor_debug_mode = "FULL_HEALTH", circular_buffer_size = -1)

  # Construct model and load initial weights
  infer_model = faster_rcnn.faster_rcnn_model(
    mode = "infer",
    num_classes = voc.Dataset.num_classes,
    allow_edge_proposals = not options.exclude_edge_proposals,
    custom_roi_pool = options.custom_roi_pool,
    detector_class_activations = not options.detector_logits
  )
  train_model = faster_rcnn.faster_rcnn_model(
    mode = "train",
    num_classes = voc.Dataset.num_classes,
    allow_edge_proposals = not options.exclude_edge_proposals,
    custom_roi_pool = options.custom_roi_pool,
    detector_class_activations = not options.detector_logits,
    l2 = 0.5 * options.weight_decay,
    dropout_probability = options.dropout
  )
  infer_model.compile()
  train_model.compile(optimizer = create_optimizer(), loss = [ None ] * len(train_model.outputs))  # losses were baked in at model construction
  if options.load_from:
    infer_model.load_weights(filepath = options.load_from, by_name = True)
    train_model.load_weights(filepath = options.load_from, by_name = True)
    print("Loaded initial weights from '%s'" % options.load_from)
  else:
    vgg16.load_imagenet_weights(infer_model)
    vgg16.load_imagenet_weights(train_model)
    print("Initialized VGG-16 layers to Keras ImageNet weights")
  vgg16.freeze_layers(infer_model, "block1_*,block2_*")
  vgg16.freeze_layers(train_model, "block1_*,block2_*") #TODO: this should be part of model construction

  # Perform mutually exclusive procedures
  if options.train:
    train(train_model = train_model, infer_model = infer_model)
  elif options.eval:
    evaluate(model = infer_model, plot = options.plot, print_average_precisions = True)
  elif options.predict:
    predict_one(model = infer_model, url = options.predict, show_image = True, output_path = None)
  elif options.predict_to_file:
    predict_one(model = infer_model, url = options.predict_to_file, show_image = False, output_path = "predictions.png")
  elif options.predict_all:
    predict_all(model = infer_model, split = options.predict_all)
  else:
    print("Nothing to do. Did you mean to use --train or --predict?")
