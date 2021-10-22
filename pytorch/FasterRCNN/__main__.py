#
# TODO:
# - Load VGG-16 from PyTorch
# - Add eval code (sampling 20% or 25% of the validation set except at the very end)
# - Investigate performance impact of generating anchor and GT maps on the fly rather than caching them
#   in the dataset code.
# - Print losses during training in tqdm bar
# - Reorg utils, separate out computations from pure utilities like nograd decorator
# - Can we load VGG-16 from Keras?
# - Support multiple batches by padding right side with zeros to a common image width
#
import argparse
import os
import torch as t
from tqdm import tqdm

from .datasets import voc
from .models.faster_rcnn import FasterRCNNModel
from . import utils
from . import visualize


def dump_anchors():
  training_data = voc.Dataset(dir = options.dataset_dir, split = options.train_split, augment = False, shuffle = False)
  if not os.path.exists(options.dump_anchors):
    os.makedirs(options.dump_anchors)
  print("Dumping anchors from '%s' to '%s'..." % (options.train_split, options.dump_anchors))
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


def train(model):
  training_data = voc.Dataset(dir = options.dataset_dir, split = options.train_split, augment = options.augment, shuffle = True)
  optimizer = t.optim.SGD(model.parameters(), lr = options.learning_rate, momentum = options.momentum)
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
  if options.save_to:
    t.save({ "epoch": epoch, "model_state_dict": model.state_dict() }, options.save_to)
    print("Saved final model weights to '%s'" % options.save_to)


def predict(model, url, show_image, output_path):
  from .datasets.image import load_image
  image_data, image, _, _ = load_image(url = url, min_dimension_pixels = 600)
  image_data = t.from_numpy(image_data).unsqueeze(dim = 0).cuda()
  scored_boxes_by_class_index = model.predict(image_data = image_data, score_threshold = 0.7)
  visualize.show_detections(
    output_path = output_path,
    show_image = show_image,
    image = image,
    scored_boxes_by_class_index = scored_boxes_by_class_index,
    class_index_to_name = voc.Dataset.class_index_to_name
  )


if __name__ == "__main__":
  parser = argparse.ArgumentParser("FasterRCNN-pytorch")
  group = parser.add_mutually_exclusive_group()
  group.add_argument("--train", action = "store_true", help = "Train model")
  group.add_argument("--predict", metavar = "url", action = "store", type = str, help = "Run inference on image and display detected boxes")
  group.add_argument("--predict-to-file", metavar = "url", action = "store", type = str, help = "Run inference on image and render detected boxes to 'detections.png'")
  parser.add_argument("--load-from", metavar = "file", action = "store", help = "Load initial model weights from file")
  parser.add_argument("--save-to", metavar = "file", action = "store", help = "Save final trained weights to file")
  parser.add_argument("--dataset-dir", metavar = "dir", action = "store", default = "../../VOCdevkit/VOC2012", help = "VOC dataset directory")
  parser.add_argument("--train-split", metavar = "name", action = "store", default = "train", help = "Dataset split to use for training")
  parser.add_argument("--epochs", metavar = "count", type = int, action = "store", default = 1, help = "Number of epochs to train for")
  parser.add_argument("--learning-rate", metavar = "value", type = float, action = "store", default = 1e-3, help = "Learning rate")
  parser.add_argument("--momentum", metavar = "value", type = float, action = "store", default = 0.9, help = "Momentum")
  parser.add_argument("--augment", action = "store_true", help = "Augment images during training using random horizontal flips")
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
    state = t.load(options.load_from)
    model.load_state_dict(state["model_state_dict"])
    print("Loaded initial weights from '%s'" % options.load_from)

  # Perform mutually exclusive procedures
  if options.train:
    train(model = model)
  elif options.predict:
    predict(model = model, url = options.predict, show_image = True, output_path = None)
  elif options.predict_to_file:
    predict(model = model, url = options.predict_to_file, show_image = False, output_path = "detections.png")
  
