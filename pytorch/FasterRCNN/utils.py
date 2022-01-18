#
# Faster R-CNN in PyTorch and TensorFlow 2 w/ Keras
# pytorch/FasterRCNN/utils.py
# Copyright 2021-2022 Bart Trzynadlowski
#
# Miscellaneous utilities.
#

import torch as t


def no_grad(func):
  def wrapper_nograd(*args, **kwargs):
    with t.no_grad():
      return func(*args, **kwargs)
  return wrapper_nograd

class CSVLog:
  """
  Logs to a CSV file.
  """
  def __init__(self, filename):
    self._filename = filename
    self._header_written = False

  def log(self, items):
    keys = list(items.keys())
    file_mode = "a" if self._header_written else "w"
    with open(self._filename, file_mode) as fp:
      if not self._header_written:
        fp.write(",".join(keys) + "\n")
        self._header_written = True
      values = [ str(value) for (key, value) in items.items() ]
      fp.write(",".join(values) + "\n")
