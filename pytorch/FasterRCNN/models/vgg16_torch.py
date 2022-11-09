#
# Faster R-CNN in PyTorch and TensorFlow 2 w/ Keras
# pytorch/FasterRCNN/models/vgg16_torch.py
# Copyright 2021-2022 Bart Trzynadlowski
#
# PyTorch implementation of the VGG-16 backbone for use as a feature extractor
# in Faster R-CNN using Torchvision's pre-trained model layers. This is an
# example of how to extract pieces of a Torchvision model. Compare with the
# custom VGG-16 implementation.
#

import torch as t
from torch import nn
from torch.nn import functional as F
import torchvision

from ..datasets import image
from .backbone import Backbone


class FeatureExtractor(nn.Module):
  def __init__(self, vgg16):
    super().__init__()

    # Get feature extractor layers
    assert len(vgg16.features) == 31 and type(vgg16.features[-1]) == nn.modules.pooling.MaxPool2d, "Torchvision VGG-16 model does not have expected architecture"
    self._layers = vgg16.features[0:-1] # get all feature extractor layers except for final one (which is a MaxPool2d)

    # Freeze first two convolutional blocks (first 4 Conv2d layers)
    i = 0
    for layer in self._layers:
      if type(layer) == nn.Conv2d and i < 4:
        layer.weight.requires_grad = False
        layer.bias.requires_grad = False
        i += 1

  def forward(self, image_data):
    return self._layers(image_data)


class PoolToFeatureVector(nn.Module):
  def __init__(self, vgg16):
    super().__init__()

    # Get classifier layers
    assert len(vgg16.classifier) == 7 and type(vgg16.classifier[-1]) == nn.modules.linear.Linear, "Torchvision VGG-16 model does not have expected architecture"
    self._layers = vgg16.classifier[0:-1] # get all classifier layers except for the final one (4096 -> 1000 ImageNet class output)

  def forward(self, rois):
    rois = rois.reshape((rois.shape[0], 512 * 7 * 7)) # flatten each RoI: (N, 512*7*7)
    return self._layers(rois)


class VGG16Backbone(Backbone):
  def __init__(self, dropout_probability):
    super().__init__()

    # Backbone properties. Image pre-processing parameters correspond to
    # Torchvision's VGG16_Weights.IMAGENET1K_V1:
    # https://pytorch.org/vision/main/models/generated/torchvision.models.vgg16.html#torchvision.models.vgg16
    self.feature_map_channels = 512
    self.feature_pixels = 16
    self.feature_vector_size = 4096
    self.image_preprocessing_params = image.PreprocessingParams(channel_order = image.ChannelOrder.RGB, scaling = 1.0 / 255.0, means = [ 0.485, 0.456, 0.406 ], stds = [ 0.229, 0.224, 0.225 ])

    # Construct model with given dropout probability and pre-loaded with ImageNet weights
    vgg16 = torchvision.models.vgg16(weights = torchvision.models.VGG16_Weights.IMAGENET1K_V1, dropout = dropout_probability)
    print("Loaded IMAGENET1K_V1 pre-trained weights for Torchvision VGG-16 backbone")

    # Feature extractor: convert input image (batch_size, channels, height, width)
    # to a feature map of shape (batch_size, 512, height // 16, width // 16)
    self.feature_extractor = FeatureExtractor(vgg16 = vgg16)

    # Conversion of pooled features to head input
    self.pool_to_feature_vector = PoolToFeatureVector(vgg16 = vgg16)

  def compute_feature_map_shape(self, image_shape):
    image_width = image_shape[-1]
    image_height = image_shape[-2]
    return (self.feature_map_channels, image_height // self.feature_pixels, image_width // self.feature_pixels)