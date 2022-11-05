#
# Faster R-CNN in PyTorch and TensorFlow 2 w/ Keras
# pytorch/FasterRCNN/models/mobilenet.py
# Copyright 2021-2022 Bart Trzynadlowski
#
# PyTorch implementation of the MobileNetV2 backbone for use as a feature
# extractor in Faster R-CNN.
#

from math import ceil
import torch as t
from torch import nn
from torch.nn import functional as F
import torchvision

from .feature_extractor import FeatureExtractor as FeatureExtractorBase


class FeatureExtractor(FeatureExtractorBase):
  def __init__(self):
    super().__init__()

    # FeatureExtractor required members
    #self.out_channels = 1280
    #self.feature_pixels = 32  # MobileNetV2 halves the input image 5 times (2^5=32)
    self.out_channels = 96
    self.feature_pixels=16

    # Construct model and pre-load with ImageNet weights
    mobilenet = torchvision.models.mobilenet_v2(weights = torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1)
    print("Loaded IMAGENET1K_V1 pre-trained weights for Torchvision MobileNetV2 feature extractor")

    # Get feature extractor layers
    #self._layers = mobilenet.features
    self._layers = mobilenet.features[0:14]

    # Freeze first two blocks
    i = 0
    for layer in self._layers:
      if i < 4:
        for name, parameter in layer.named_parameters():
          parameter.required_grad = False
      i += 1

  def compute_feature_map_shape(self, image_shape):
    image_width = image_shape[-1]
    image_height = image_shape[-2]
    return (self.out_channels, ceil(image_height / self.feature_pixels), ceil(image_width / self.feature_pixels))

  def forward(self, image_data):
    """
    Converts input images into feature maps using MobileNetV2 feature extractor
    layers.

    Parameters
    ----------
    image_data : torch.Tensor
      A tensor of shape (batch_size, channels, height, width) representing
      images normalized as required by Torchvision's MobileNetV2
      implementation.

    Returns
    -------
    torch.Tensor
      Feature map of shape (batch_size, 512, height // 32, width // 32).
    """
    return self._layers(image_data)