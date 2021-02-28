from . import utils
from .dataset import VOC

import argparse
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras
import time

if __name__ == "__main__":
  parser = argparse.ArgumentParser("FasterRCNN")
  parser.add_argument("--dataset-dir", metavar = "path", type = str, action = "store", default = "\\projects\\voc\\vocdevkit\\voc2012", help = "Dataset directory")
  options = parser.parse_args()

  voc = VOC(dataset_dir = options.dataset_dir)
  print(voc.index_to_class_name)