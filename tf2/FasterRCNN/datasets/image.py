#
# Faster R-CNN in PyTorch and TensorFlow 2 w/ Keras
# tf2/FasterRCNN/datasets/image.py
# Copyright 2021-2022 Bart Trzynadlowski
#
# Image loading and pre-processing.
#

import imageio
from PIL import Image
import numpy as np

def _compute_scale_factor(original_width, original_height, min_dimension_pixels):
  if not min_dimension_pixels:
    return 1.0
  if original_width > original_height:
    scale_factor = min_dimension_pixels / original_height
  else:
    scale_factor = min_dimension_pixels / original_width
  return scale_factor

def _preprocess_vgg16(image_data):
  image_data = image_data[:, :, ::-1]           # RGB -> BGR
  image_data[:, :, 0] -= 103.939                # ImageNet B mean
  image_data[:, :, 1] -= 116.779                # ImageNet G mean
  image_data[:, :, 2] -= 123.680                # ImageNet R mean 
  return image_data

def load_image(url, min_dimension_pixels = None, horizontal_flip = False):
  """
  Loads and preprocesses an image for use with VGG-16, which consists of
  converting RGB to BGR and subtracting ImageNet dataset means from each
  component. The image can be resized so that the minimum dimension is a
  defined size, as recommended by Faster R-CNN. 

  Parameters
  ----------
  url : str
    URL (local or remote file) to load.
  min_dimension_pixels : int
    If not None, specifies the size in pixels of the smaller side of the image.
    The other side is scaled proportionally.
  horizontal_flip : bool
    Whether to flip the image horizontally.

  Returns
  -------
  np.ndarray, PIL.Image, float, Tuple[int, int, int]
    Image pixels as float32, shaped as (channels, height, width); an image
    object suitable for drawing and visualization; scaling factor applied to
    the image dimensions; and the original image shape.
  """
  data = imageio.imread(url, pilmode = "RGB")
  image = Image.fromarray(data, mode = "RGB")
  original_width, original_height = image.width, image.height
  if horizontal_flip:
    image = image.transpose(method = Image.FLIP_LEFT_RIGHT)
  if min_dimension_pixels is not None:
    scale_factor = _compute_scale_factor(original_width = image.width, original_height = image.height, min_dimension_pixels = min_dimension_pixels)
    width = int(image.width * scale_factor)
    height = int(image.height * scale_factor)
    image = image.resize((width, height), resample = Image.BILINEAR)
  else:
    scale_factor = 1.0
  image_data = np.array(image).astype(np.float32)
  image_data = _preprocess_vgg16(image_data = image_data)
  return image_data, image, scale_factor, (image_data.shape[0], original_height, original_width)
