#
# Faster R-CNN in PyTorch and TensorFlow 2 w/ Keras
# pytorch/FasterRCNN/__main__.py
# Copyright 2021-2022 Bart Trzynadlowski
#
# Main module for the PyTorch implementation of Faster R-CNN. Run this from the
# root directory, e.g.:
#
# python -m pytorch.FasterRCNN --help
#

#
# TODO
# ----
# - Support multiple batches by padding right side with zeros to a common image
#   width
# - Support for loading Keras checkpoints (particularly if using custom RoI
#   pooling, which should be almost the same as PyTorch's RoI pool layer)
#

import argparse
import os
import torch as t
from tqdm import tqdm

from .datasets import voc
from .models.faster_rcnn import FasterRCNNModel
from .models import vgg16
from .models import vgg16_torch
from .models import resnet
from .statistics import TrainingStatistics
from .statistics import PrecisionRecallCurveCalculator
from . import state
from . import utils
from . import visualize


def render_anchors(backbone):
  training_data = voc.Dataset(
    image_preprocessing_params = backbone.image_preprocessing_params,
    compute_feature_map_shape_fn = backbone.compute_feature_map_shape,
    feature_pixels = backbone.feature_pixels,
    dir = options.dataset_dir,
    split = options.train_split,
    augment = False,
    shuffle = False
  )
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

def evaluate(model, eval_data = None, num_samples = None, plot = False, print_average_precisions = False):
  if eval_data is None:
    eval_data = voc.Dataset(
      image_preprocessing_params = model.backbone.image_preprocessing_params,
      compute_feature_map_shape_fn = model.backbone.compute_feature_map_shape,
      feature_pixels = model.backbone.feature_pixels,
      dir = options.dataset_dir,
      split = options.eval_split,
      augment = False,
      shuffle = False
    )
  if num_samples is None:
    num_samples = eval_data.num_samples
  precision_recall_curve = PrecisionRecallCurveCalculator()
  i = 0
  print("Evaluating '%s'..." % eval_data.split)
  for sample in tqdm(iterable = iter(eval_data), total = num_samples):
    scored_boxes_by_class_index = model.predict(
      image_data = t.from_numpy(sample.image_data).unsqueeze(dim = 0).cuda(),
      score_threshold = 0.05  # lower threshold for evaluation
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

def create_optimizer(model):
  params = []
  for key, value in dict(model.named_parameters()).items():
    if not value.requires_grad:
      continue
    if "weight" in key:
      params += [{ "params": [value], "weight_decay": options.weight_decay }]
  return t.optim.SGD(params, lr = options.learning_rate, momentum = options.momentum)

def enable_cuda_memory_profiler(model):
  from pytorch.FasterRCNN import profile
  import sys
  import threading
  memory_profiler = profile.CUDAMemoryProfiler([ model ], filename = "cuda_memory.txt")
  sys.settrace(memory_profiler)
  threading.settrace(memory_profiler)

def train(model):
  if options.profile_cuda_memory:
    enable_cuda_memory_profiler(model = model)
  if options.load_from:
    initial_weights = options.load_from
  else:
    if options.backbone == "vgg16":  # "vgg16" is our hand-implemented backbone and the only one without built-in default weights
      initial_weights = "none"
    else:
      initial_weights = "IMAGENET1K_V1"
  print("Training Parameters")
  print("-------------------")
  print("Initial weights   : %s" % initial_weights)
  print("Dataset           : %s" % options.dataset_dir)
  print("Training split    : %s" % options.train_split)
  print("Evaluation split  : %s" % options.eval_split)
  print("Backbone          : %s" % options.backbone)
  print("Epochs            : %d" % options.epochs)
  print("Learning rate     : %f" % options.learning_rate)
  print("Momentum          : %f" % options.momentum)
  print("Weight decay      : %f" % options.weight_decay)
  print("Dropout           : %f" % options.dropout)
  print("Augmentation      : %s" % ("disabled" if options.no_augment else "enabled"))
  print("Edge proposals    : %s" % ("excluded" if options.exclude_edge_proposals else "included"))
  print("CSV log           : %s" % ("none" if not options.log_csv else options.log_csv))
  print("Checkpoints       : %s" % ("disabled" if not options.checkpoint_dir else options.checkpoint_dir))
  print("Final weights file: %s" % ("none" if not options.save_to else options.save_to))
  print("Best weights file : %s" % ("none" if not options.save_best_to else options.save_best_to))
  training_data = voc.Dataset(
    dir = options.dataset_dir,
    split = options.train_split,
    image_preprocessing_params = model.backbone.image_preprocessing_params,
    compute_feature_map_shape_fn = model.backbone.compute_feature_map_shape,
    feature_pixels = model.backbone.feature_pixels,
    augment = not options.no_augment,
    shuffle = True,
    cache = options.cache_images
  )
  eval_data = voc.Dataset(
    dir = options.dataset_dir,
    split = options.eval_split,
    image_preprocessing_params = model.backbone.image_preprocessing_params,
    compute_feature_map_shape_fn = model.backbone.compute_feature_map_shape,
    feature_pixels = model.backbone.feature_pixels,
    augment = False,
    shuffle = False,
    cache = False
  )
  optimizer = create_optimizer(model = model)
  if options.checkpoint_dir and not os.path.exists(options.checkpoint_dir):
    os.makedirs(options.checkpoint_dir)
  if options.log_csv:
    csv = utils.CSVLog(options.log_csv)
  if options.save_best_to:
    best_weights_tracker = state.BestWeightsTracker(filepath = options.save_best_to)
  for epoch in range(1, 1 + options.epochs):
    print("Epoch %d/%d" % (epoch, options.epochs))
    stats = TrainingStatistics()
    progbar = tqdm(iterable = iter(training_data), total = training_data.num_samples, postfix = stats.get_progbar_postfix())
    for sample in progbar:
      loss = model.train_step(  # don't retain any tensors we don't need (helps memory usage)
        optimizer = optimizer,
        image_data = t.from_numpy(sample.image_data).unsqueeze(dim = 0).cuda(),
        anchor_map = sample.anchor_map,
        anchor_valid_map = sample.anchor_valid_map,
        gt_rpn_map = t.from_numpy(sample.gt_rpn_map).unsqueeze(dim = 0).cuda(),
        gt_rpn_object_indices = [ sample.gt_rpn_object_indices ],
        gt_rpn_background_indices = [ sample.gt_rpn_background_indices ],
        gt_boxes = [ sample.gt_boxes ]
      )
      stats.on_training_step(loss = loss)
      progbar.set_postfix(stats.get_progbar_postfix())
    last_epoch = epoch == options.epochs
    mean_average_precision = evaluate(
      model = model,
      eval_data = eval_data,
      num_samples = options.periodic_eval_samples,
      plot = False,
      print_average_precisions = False
    )
    if options.checkpoint_dir:
      checkpoint_file = os.path.join(options.checkpoint_dir, "checkpoint-epoch-%d-mAP-%1.1f.pth" % (epoch, mean_average_precision))
      t.save({ "epoch": epoch, "model_state_dict": model.state_dict() }, checkpoint_file)
      print("Saved model checkpoint to '%s'" % checkpoint_file)
    if options.log_csv:
      log_items = {
        "epoch": epoch,
        "learning_rate": options.learning_rate,
        "momentum": options.momentum,
        "weight_decay": options.weight_decay,
        "dropout": options.dropout,
        "mAP": mean_average_precision
      }
      log_items.update(stats.get_progbar_postfix())
      csv.log(log_items)
    if options.save_best_to:
      best_weights_tracker.on_epoch_end(model = model, epoch = epoch, mAP = mean_average_precision)
  if options.save_to:
    t.save({ "epoch": epoch, "model_state_dict": model.state_dict() }, options.save_to)
    print("Saved final model weights to '%s'" % options.save_to)
  if options.save_best_to:
    best_weights_tracker.save_best_weights(model = model)
  print("Evaluating %s model on all samples in '%s'..." % (("best" if options.save_best_to else "final"), options.eval_split))  # evaluate final or best model on full dataset
  evaluate(
    model = model,
    eval_data = eval_data,
    num_samples = eval_data.num_samples,  # use all samples
    plot = options.plot,
    print_average_precisions = True
  )

def predict(model, image_data, image, show_image, output_path):
  image_data = t.from_numpy(image_data).unsqueeze(dim = 0).cuda()
  scored_boxes_by_class_index = model.predict(image_data = image_data, score_threshold = 0.7)
  visualize.show_detections(
    output_path = output_path,
    show_image = show_image,
    image = image,
    scored_boxes_by_class_index = scored_boxes_by_class_index,
    class_index_to_name = voc.Dataset.class_index_to_name
  )

def predict_one(model, url, show_image, output_path):
  from .datasets import image
  image_data, image_obj, _, _ = image.load_image(url = url, preprocessing = model.backbone.image_preprocessing_params, min_dimension_pixels = 600)
  predict(model = model, image_data = image_data, image = image_obj, show_image = show_image, output_path = output_path)

def predict_all(model, split):
  dirname = "predictions_" + split
  if not os.path.exists(dirname):
    os.makedirs(dirname)
  print("Rendering predictions from '%s' set to '%s'..." % (split, dirname))
  dataset = voc.Dataset(
    dir = options.dataset_dir,
    split = split,
    image_preprocessing_params = model.backbone.image_preprocessing_params,
    compute_feature_map_shape_fn = model.backbone.compute_feature_map_shape,
    feature_pixels = model.backbone.feature_pixels,
    augment = False,
    shuffle = False
  )
  for sample in iter(dataset):
    output_path = os.path.join(dirname, os.path.splitext(os.path.basename(sample.filepath))[0] + ".png")
    predict(model = model, image_data = sample.image_data, image = sample.image, show_image = False, output_path = output_path)

if __name__ == "__main__":
  parser = argparse.ArgumentParser("FasterRCNN")
  group = parser.add_mutually_exclusive_group()
  group.add_argument("--train", action = "store_true", help = "Train model")
  group.add_argument("--eval", action = "store_true", help = "Evaluate model")
  group.add_argument("--predict", metavar = "url", action = "store", type = str, help = "Run inference on image and display detected boxes")
  group.add_argument("--predict-to-file", metavar = "url", action = "store", type = str, help = "Run inference on image and render detected boxes to 'predictions.png'")
  group.add_argument("--predict-all", metavar = "name", action = "store", type = str, help = "Run inference on all images in the specified dataset split and write to directory 'predictions_${split}'")
  parser.add_argument("--load-from", metavar = "file", action = "store", help = "Load initial model weights from file")
  parser.add_argument("--backbone", metavar = "model", action = "store", default = "vgg16", help = "Backbone model for feature extraction and classification")
  parser.add_argument("--save-to", metavar = "file", action = "store", help = "Save final trained weights to file")
  parser.add_argument("--save-best-to", metavar = "file", action = "store", help = "Save best weights (highest mean average precision) to file")
  parser.add_argument("--dataset-dir", metavar = "dir", action = "store", default = "VOCdevkit/VOC2007", help = "VOC dataset directory")
  parser.add_argument("--train-split", metavar = "name", action = "store", default = "trainval", help = "Dataset split to use for training")
  parser.add_argument("--eval-split", metavar = "name", action = "store", default = "test", help = "Dataset split to use for evaluation")
  parser.add_argument("--cache-images", action = "store_true", help = "Cache images during training (requires ample CPU memory)")
  parser.add_argument("--periodic-eval-samples", metavar = "count", action = "store", default = 1000, help = "Number of samples to use during evaluation after each epoch")
  parser.add_argument("--checkpoint-dir", metavar = "dir", action = "store", help = "Save checkpoints after each epoch to the given directory")
  parser.add_argument("--plot", action = "store_true", help = "Plots the average precision of each class after evaluation (use with --train or --eval)")
  parser.add_argument("--log-csv", metavar = "file", action = "store", help = "Log training metrics to CSV file")
  parser.add_argument("--epochs", metavar = "count", type = int, action = "store", default = 1, help = "Number of epochs to train for")
  parser.add_argument("--learning-rate", metavar = "value", type = float, action = "store", default = 1e-3, help = "Learning rate")
  parser.add_argument("--momentum", metavar = "value", type = float, action = "store", default = 0.9, help = "Momentum")
  parser.add_argument("--weight-decay", metavar = "value", type = float, action = "store", default = 5e-4, help = "Weight decay")
  parser.add_argument("--dropout", metavar = "probability", type = float, action = "store", default = 0.0, help = "Dropout probability after each of the two fully-connected detector layers")
  parser.add_argument("--no-augment", action = "store_true", help = "Disable image augmentation (random horizontal flips) during training")
  parser.add_argument("--exclude-edge-proposals", action = "store_true", help = "Exclude proposals generated at anchors spanning image edges from being passed to detector stage")
  parser.add_argument("--dump-anchors", metavar = "dir", action = "store", help = "Render out all object anchors and ground truth boxes from the training set to a directory")
  parser.add_argument("--profile-cuda-memory", action = "store_true", help = "Profile CUDA memory usage and write output to 'cuda_memory.txt'")
  options = parser.parse_args()

  # Validate backbone model
  valid_backbones = [ "vgg16", "vgg16-torch", "resnet50", "resnet101", "resnet152" ]
  assert options.backbone in valid_backbones, "--backbone must be one of: " + ", ".join(valid_backbones)
  if options.dropout != 0:
    assert options.backbone == "vgg16" or options.backbone == "vgg16-torch", "--dropout can only be used with VGG-16 backbones"
  if options.backbone == "vgg16":
    backbone = vgg16.VGG16Backbone(dropout_probability = options.dropout)
  elif options.backbone == "vgg16-torch":
    backbone = vgg16_torch.VGG16Backbone(dropout_probability = options.dropout)
  elif options.backbone == "resnet50":
    backbone = resnet.ResNetBackbone(architecture = resnet.Architecture.ResNet50)
  elif options.backbone == "resnet101":
    backbone = resnet.ResNetBackbone(architecture = resnet.Architecture.ResNet101)
  elif options.backbone == "resnet152":
    backbone = resnet.ResNetBackbone(architecture = resnet.Architecture.ResNet152)

  # Perform optional procedures
  if options.dump_anchors:
    render_anchors(backbone = backbone)

  # Construct model and load initial weights
  model = FasterRCNNModel(
    num_classes = voc.Dataset.num_classes,
    backbone = backbone,
    allow_edge_proposals = not options.exclude_edge_proposals
  ).cuda()
  if options.load_from:
    state.load(model = model, filepath = options.load_from)

  # Perform mutually exclusive procedures
  if options.train:
    train(model = model)
  elif options.eval:
    evaluate(model = model, plot = options.plot, print_average_precisions = True)
  elif options.predict:
    predict_one(model = model, url = options.predict, show_image = True, output_path = None)
  elif options.predict_to_file:
    predict_one(model = model, url = options.predict_to_file, show_image = False, output_path = "predictions.png")
  elif options.predict_all:
    predict_all(model = model, split = options.predict_all)
  elif not options.dump_anchors:
    print("Nothing to do. Did you mean to use --train or --predict?")