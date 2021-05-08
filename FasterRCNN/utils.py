#
# FasterRCNN for Keras
# Copyright 2021 Bart Trzynadlowski
#
# utils.py
#
# Miscellaneous utilities.
#

import argparse
import imageio
from PIL import Image
import numpy as np
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

def compute_new_image_dimensions(original_width, original_height, min_dimension_pixels):
  if not min_dimension_pixels:
    return (original_width, original_height)
  if original_width > original_height:
    new_width = (original_width / original_height) * min_dimension_pixels
    new_height = min_dimension_pixels
  else:
    new_height = (original_height / original_width) * min_dimension_pixels
    new_width = min_dimension_pixels
  return (int(new_width), int(new_height))

def load_image(url, min_dimension_pixels):
  data = imageio.imread(url, pilmode = "RGB")
  image = Image.fromarray(data, mode = "RGB")
  width, height = compute_new_image_dimensions(original_width = image.width, original_height = image.height, min_dimension_pixels = min_dimension_pixels)
  image = image.resize((width, height), resample = Image.BILINEAR)
  return image

def load_image_data_vgg16(url, min_dimension_pixels):
  """
  Loads an image and returns a NumPy tensor of shape (height,width,3), pre-
  processed for VGG-16: BGR order and ImageNet component-wise mean pre-
  subtracted.

  Parameters
  ----------
  url : str
    URL or path of file to load.
  min_dimension_pixels: int
    New size of the image's minimum dimension. The other dimension will be
    scaled proportionally. Bilinear sampling is used.

  Returns
  -------
  np.ndarray
    Image data.
  """
  image = np.array(load_image(url = url, min_dimension_pixels = min_dimension_pixels))
  return tf.keras.applications.vgg16.preprocess_input(x = image)

def freeze_layers(model, layers):
  """
  Sets specified layers in a model to non-trainable. Layers may be specified as
  a string of comma-separated layer names or as a list of layer names.
  """
  frozen_layers = []
  if type(layers) == str:
    frozen_layers = [ frozen_layer.strip() for frozen_layer in layers.split(",") ]
  elif type(layers) == list:
    frozen_layers = layers
  elif layers != None:
    raise RuntimeError("freeze_layers: 'layers' must be a string or a list of strings")
  for layer in model.layers:
    if layer.name in frozen_layers:
      layer.trainable = False
      frozen_layers.remove(layer.name)
  if len(frozen_layers) > 0:
    raise RuntimeError("Cannot freeze missing layers: %s" % ", ".join(frozen_layers))

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
