#
# TODO:
# - Test weight decay by making it very large
# - Investigate performance impact of generating anchor and GT maps on the fly rather than caching them
#   in the dataset code. If no impact, just calculate them when needed.
# - Print other statistics
# - Add automatic checkpoints
# - Print losses during training in tqdm bar
# - Reorg utils, separate out computations from pure utilities like nograd decorator
# - Support multiple batches by padding right side with zeros to a common image width
# - Add tqdm to render_anchors, predict_all, etc.
# - Move anchor code from dataset/ to models/
# - Add dropout and regularization
# - Once a new Keras implementation exists, add support for loading complete state from h5
#
import argparse
import os
import torch as t
from tqdm import tqdm

from .datasets import voc
from .models.faster_rcnn import FasterRCNNModel
from .statistics import PrecisionRecallCurveCalculator
from . import state
from . import utils
from . import visualize


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

def evaluate(model, eval_data = None, num_samples = None, plot = False):
  if eval_data is None:
    eval_data = voc.Dataset(dir = options.dataset_dir, split = options.eval_split, augment = False, shuffle = False)
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
  print("Mean Average Precision = %1.2f%%" % (100.0 * precision_recall_curve.compute_mean_average_precision()))
  if plot:
    precision_recall_curve.plot_average_precisions(class_index_to_name = voc.Dataset.class_index_to_name)

def create_optimizer(model):
  params = []
  for key, value in dict(model.named_parameters()).items():
    if not value.requires_grad:
      continue
    if "weight" in key:
      params += [{ "params": [value], "weight_decay": options.weight_decay }]
  return t.optim.SGD(params, lr = options.learning_rate, momentum = options.momentum)

def train(model):
  print("Training Parameters")
  print("-------------------")
  print("Epochs       : %d" % options.epochs)
  print("Learning rate: %f" % options.learning_rate)
  print("Momentum     : %f" % options.momentum)
  print("Weight decay : %f" % options.weight_decay)
  print("Augmentation : %s" % ("disabled" if options.no_augment else "enabled"))
  training_data = voc.Dataset(dir = options.dataset_dir, split = options.train_split, augment = not options.no_augment, shuffle = True, cache = not options.no_cache)
  eval_data = voc.Dataset(dir = options.dataset_dir, split = options.eval_split, augment = False, shuffle = False, cache = False)
  optimizer = create_optimizer(model = model)
  for epoch in range(1, 1 + options.epochs):
    print("Epoch %d/%d" % (epoch, options.epochs))
    for sample in tqdm(iterable = iter(training_data), total = training_data.num_samples):
      loss, rpn_score_map, rpn_regressions_map, classes, regressions, gt_classes, gt_regressions = model.train_step(
          optimizer = optimizer,
          image_data = t.from_numpy(sample.image_data).unsqueeze(dim = 0).cuda(),
          anchor_map = sample.anchor_map,
          anchor_valid_map = sample.anchor_valid_map,
          gt_rpn_map = t.from_numpy(sample.gt_rpn_map).unsqueeze(dim = 0).cuda(),
          gt_rpn_object_indices = [ sample.gt_rpn_object_indices ],
          gt_rpn_background_indices = [ sample.gt_rpn_background_indices ],
          gt_boxes = [ sample.gt_boxes ]
        )
    last_epoch = epoch == options.epochs
    evaluate(
      model = model,
      eval_data = eval_data,
      num_samples = None if last_epoch else options.periodic_eval_samples, # use full number of samples at last epoch
      plot = options.plot if last_epoch else False
    )
  if options.save_to:
    t.save({ "epoch": epoch, "model_state_dict": model.state_dict() }, options.save_to)
    print("Saved final model weights to '%s'" % options.save_to)

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
  from .datasets.image import load_image
  image_data, image, _, _ = load_image(url = url, min_dimension_pixels = 600)
  predict(model = model, image_data = image_data, image = image, show_image = show_image, output_path = output_path)

def predict_all(model, split):
  dirname = "predictions_" + split
  if not os.path.exists(dirname):
    os.makedirs(dirname)
  print("Rendering predictions from '%s' set to '%s'..." % (split, dirname))
  dataset = voc.Dataset(dir = options.dataset_dir, split = split, augment = False, shuffle = False)
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
  parser.add_argument("--save-to", metavar = "file", action = "store", help = "Save final trained weights to file")
  parser.add_argument("--dataset-dir", metavar = "dir", action = "store", default = "../../VOCdevkit/VOC2007", help = "VOC dataset directory")
  parser.add_argument("--train-split", metavar = "name", action = "store", default = "trainval", help = "Dataset split to use for training")
  parser.add_argument("--eval-split", metavar = "name", action = "store", default = "test", help = "Dataset split to use for evaluation")
  parser.add_argument("--periodic-eval-samples", metavar = "count", action = "store", default = 1000, help = "Number of samples to use during evaluation after each epoch")
  parser.add_argument("--no-cache", action = "store_true", help = "Disable image caching during training (reduces memory usage)")
  parser.add_argument("--plot", action = "store_true", help = "Plots the average precision of each class after evaluation (use with --train or --eval)")
  parser.add_argument("--epochs", metavar = "count", type = int, action = "store", default = 1, help = "Number of epochs to train for")
  parser.add_argument("--learning-rate", metavar = "value", type = float, action = "store", default = 1e-3, help = "Learning rate")
  parser.add_argument("--momentum", metavar = "value", type = float, action = "store", default = 0.9, help = "Momentum")
  parser.add_argument("--weight-decay", metavar = "value", type = float, action = "store", default = 0.0, help = "Weight decay")
  parser.add_argument("--no-augment", action = "store_true", help = "Disable image augmentation (random horizontal flips) during training")
  parser.add_argument("--exclude-edge-proposals", action = "store_true", help = "Exclude proposals generated at anchors spanning image edges from being passed to detector stage")
  parser.add_argument("--dump-anchors", metavar = "dir", action = "store", help = "Render out all object anchors and ground truth boxes from the training set to a directory")
  #TODO: proposal batch
  #TODO: anchor minibatch
  options = parser.parse_args()

  # Perform optional procedures
  if options.dump_anchors:
    dump_anchors()

  # Construct model and load initial weights
  model = FasterRCNNModel(num_classes = voc.Dataset.num_classes, allow_edge_proposals = not options.exclude_edge_proposals).cuda()
  if options.load_from:
    state.load(model = model, filepath = options.load_from)

  # Perform mutually exclusive procedures
  if options.train:
    train(model = model)
  elif options.eval:
    evaluate(model = model, plot = options.plot)
  elif options.predict:
    predict_one(model = model, url = options.predict, show_image = True, output_path = None)
  elif options.predict_to_file:
    predict_one(model = model, url = options.predict_to_file, show_image = False, output_path = "predictions.png")
  elif options.predict_all:
    predict_all(model = model, split = options.predict_all)
  else:
    print("Nothing to do. Did you mean to use --train or --predict?")
  
