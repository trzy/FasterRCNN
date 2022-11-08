#
# Faster R-CNN in PyTorch and TensorFlow 2 w/ Keras
# pytorch/FasterRCNN/models/backbone.py
# Copyright 2021-2022 Bart Trzynadlowski
#
# Backbone base class, for wrapping backbone models that provide feature
# extraction and pooled feature reduction layers from the classifier
# stages.
#
# The backbone in Faster R-CNN is used in two places:
#
#   1. In Stage 1 as the feature extractor. Given an input image, a feature map
#      is produced that is then passed into both the RPN and detector stages.
#   2. In Stage 3, the detector, proposal regions are pooled and cropped from
#      the feature map (to produce RoIs) and fed into the detector layers,
#      which perform classification and bounding box regression. Each RoI must
#      first be converted into a linear feature vector. With VGG-16, for
#      example, the fully-connected layers following the convolutional layers
#      and preceding the classifier layer, are used to do this.
#

import torch as t
from torch import nn
from torch.nn import functional as F
import torchvision

from ..datasets import image


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
    self.image_preprocessing_params = image.PreprocessingParams(channel_order = image.ChannelOrder.BGR, scaling = 1.0, means = [ 103.939, 116.779, 123.680 ], stds = [ 1, 1, 1 ])

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