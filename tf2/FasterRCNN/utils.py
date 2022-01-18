#
# Faster R-CNN in PyTorch and TensorFlow 2 w/ Keras
# tf2/FasterRCNN/utils.py
# Copyright 2021-2022 Bart Trzynadlowski
#
# Miscellaneous utilities.
#

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

