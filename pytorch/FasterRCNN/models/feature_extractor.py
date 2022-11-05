#
# Faster R-CNN in PyTorch and TensorFlow 2 w/ Keras
# pytorch/FasterRCNN/models/backbone.py
# Copyright 2021-2022 Bart Trzynadlowski
#
# Backbone base class, for wrapping backbone models that provide feature
# extraction and pooled feature reduction layers from the classifier
# stages.
#

import torch as t
from torch import nn
from torch.nn import functional as F
import torchvision


class Backbone:
  """
  Backbone base class. When overriding, ensure all members and methods are
  defined.
  """
  def __init__(self):
    # Required properties
    self.feature_map_channels = 0     # feature map channels
    self.feature_pixels = 0           # feature size in pixels, N: each feature map cell corresponds to an NxN area on original image
    self.feature_vector_size = 0      # length of linear feature vector after pooling and just before being passed to detector heads

    # Required members
    self.feature_extractor = None       # nn.Module converting input image (batch_size, channels, width, height) -> (batch_size, feature_map_channels, W, H)
    self.pool_to_feature_vector = None  # nn.Module converting RoIs (N, feature_map_channels, 7, 7) -> (N, feature_vector_size)

  def compute_feature_map_shape(self, image_shape):
    """
    Computes the shape of the feature extractor output given an input image
    shape. This is used primarily for anchor generation and depends entirely on
    the architecture of the backbone.

    Parameters
    ----------
    image_shape : Tuple[int, int, int]
      Shape of the input image, (channels, height, width). Only the last two
      dimensions are relevant, allowing image_shape to be either the shape
      of a single image or the entire batch.

    Returns
    -------
    Tuple[int, int, int]
      Shape of the feature map produced by the feature extractor,
      (feature_map_channels, feature_map_height, feature_map_width).
    """
    return image_shape[-3:]