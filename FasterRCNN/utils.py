#
# VGG16
# utils.py
# Copyright 2020-2021 Bart Trzynadlowski
#
# Miscellaneous utilities.
#

import argparse
import tensorflow as tf
from tensorflow.keras import backend as K

def positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue

def int_range_str(value):
  parts = value.split(",")
  if len(parts) <= 0 or len(parts) > 2:
    raise argparse.ArgumentTypeError("Range argument must be one or two comma-separated values")
  if len(parts) == 1:
    parts = [ parts[0], parts[0] ]
  values = [ int(parts[0]), int(parts[1]) ]
  values = [ min(values), max(values) ]
  return values

class CSVLogCallback(tf.keras.callbacks.Callback):
  """
  Keras callback to log metrics along with epoch number and learning rate to a
  CSV file.
  """
  def __init__(self, filename, log_epoch_number = False, log_learning_rate = False):
    super().__init__()
    self.filename = filename
    self.header_written = False
    self.log_epoch_number = log_epoch_number
    self.log_learning_rate = log_learning_rate
    self.epoch_number = 0

  def on_epoch_end(self, epoch, logs = None):
    self.epoch_number += 1
    keys = list(logs.keys())
    extra_keys = []
    if self.log_epoch_number:
      extra_keys.append("epoch")
    if self.log_learning_rate:
      extra_keys.append("learning_rate")
    keys = extra_keys + keys
    file_mode = "a" if self.header_written else "w"
    with open(self.filename, file_mode) as fp:
      if not self.header_written:
        fp.write(",".join(keys) + "\n")
        self.header_written = True
      values = [ str(value) for (key, value) in logs.items() ]
      extra_values = []
      if self.log_epoch_number:
        extra_values.append(str(self.epoch_number))
      if self.log_learning_rate:
        learning_rate = K.eval(self.model.optimizer.lr)
        extra_values.append(str(learning_rate))
      values = extra_values + values
      fp.write(",".join(values) + "\n")
