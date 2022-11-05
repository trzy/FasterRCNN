#
# https://github.com/jwyang/faster-rcnn.pytorch/blob/f9d984d27b48a067b29792932bcb5321a39c1f09/lib/model/faster_rcnn/resnet.py
#
# Backbone needs to be split into feature extractor and classifier
# For ResNet, feature extractor is up through layer 3, and output will be (batch, 1024, H, W).
# This can be fed to RoI pool and will end up producing (N, 1024, 7, 7)
# This in turn can be fed to the classifier layer, which for ResNet is just layer 4 and will output (N, 2048, 4, 4).
# The last two dimensions must be eliminated by averaging: .mean(3).mean(2)
# This produces (N, 2048) which can be fed into detector classifier and regressor layers (inputs to this must be adjusted from 4096->2048)

#
# TODO: rather than mean, can we just flatten?

# TODO: FeatureExtractor -> Backbone, since it now includes tail

#TODO: dropout only for VGG-16

#TODO: make note about feature map -> pixel conversion in anchors.py

#TODO: explanation in vgg16.py of what is going on here



#
# Faster R-CNN in PyTorch and TensorFlow 2 w/ Keras
# pytorch/FasterRCNN/models/resnet.py
# Copyright 2021-2022 Bart Trzynadlowski
#
# PyTorch implementation of the ResNet backbone for use as a feature
# extractor in Faster R-CNN.
#

from math import ceil
import torch as t
from torch import nn
from torch.nn import functional as F
import torchvision

from .feature_extractor import Backbone


class PoolToFeatureVector(nn.Module):
  def __init__(self, resnet):
    super().__init__()
    self._layer4 = resnet.layer4

  def forward(self, rois):
    y = self._layer4(rois)  # (N, 1024, 7, 7) -> (N, 2048, 4, 4)
    y = y.mean(-1).mean(-1) # use mean to remove last two dimensions -> (N, 2048)
    return y


class FeatureExtractor(nn.Module):
  def __init__(self, resnet):
    super().__init__()

    # Feature extractor layers
    self._feature_extractor = nn.Sequential(
      resnet.conv1,
      resnet.bn1,
      resnet.relu,
      resnet.maxpool,
      resnet.layer1,
      resnet.layer2,
      resnet.layer3
    )

    # Freeze first two layers (conv1, bn1)
    i = 0
    for layer in self._feature_extractor:
      if i < 2:
        for name, parameter in layer.named_parameters():
          parameter.required_grad = False
      i += 1

    # Freeze first two blocks
    i = 4
    while i < 6:
      for name, parameter in self._feature_extractor[i].named_parameters():
          parameter.required_grad = False
      i += 1

  def forward(self, image_data):
    return self._feature_extractor(image_data)


class ResNetBackbone(Backbone):
  def __init__(self):
    super().__init__()

    # Backbone properties
    self.feature_map_channels = 1024  # feature extractor output channels
    self.feature_pixels = 16          # ResNet feature maps are 1/16th of the original image size, similar to VGG-16 feature extractor
    self.feature_vector_size = 2048   # linear feature vector size after pooling

    # Construct model and pre-load with ImageNet weights
    resnet = torchvision.models.resnet50(weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    print("Loaded IMAGENET1K_V1 pre-trained weights for Torchvision ResNet50 feature extractor")

    # Feature extractor: given image data of shape (batch_size, channels, height, width),
    # produces a feature map of shape (batch_size, 1024, ceil(height/16), ceil(width/16))
    self.feature_extractor = FeatureExtractor(resnet = resnet)

    # Conversion of pooled features to head input
    self.pool_to_feature_vector = PoolToFeatureVector(resnet = resnet)

  def compute_feature_map_shape(self, image_shape):
    """
    Computes feature map shape given input image shape. Unlike VGG-16, ResNet
    convolutional layers use padding and the resultant dimensions are therefore
    not simply an integral division by 16. The calculation here works well
    enough but it is not guaranteed that the simple conversion of feature map
    coordinates to input image pixel coordinates in anchors.py is absolutely
    correct.

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
    image_width = image_shape[-1]
    image_height = image_shape[-2]
    return (self.feature_map_channels, ceil(image_height / self.feature_pixels), ceil(image_width / self.feature_pixels))