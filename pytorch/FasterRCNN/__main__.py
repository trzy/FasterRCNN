import argparse
import os

from .datasets import voc
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


if __name__ == "__main__":
  parser = argparse.ArgumentParser("FasterRCNN-pytorch")
  parser.add_argument("--dataset-dir", metavar = "dir", action = "store", default = "../../VOCdevkit/VOC2012", help = "VOC dataset directory")
  parser.add_argument("--train-split", metavar = "name", action = "store", default = "train", help = "Dataset split to use for training")
  parser.add_argument("--dump-anchors", metavar = "dir", action = "store", help = "Render out all object anchors and ground truth boxes from the training set to a directory")
  options = parser.parse_args()

  if options.dump_anchors:
    dump_anchors()

