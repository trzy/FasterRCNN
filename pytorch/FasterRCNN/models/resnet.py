#TODO:
# - Cleanup, particularly of the train() methods and add an explanation for what is going on, tying it to paper if possible

#
# This repo demonstrates how to use a ResNet backbone. In particular, freezing of the batchnorm layers is very important.
# TODO: does the ResNet paper, which mentions its deployment in Faster R-CNN, mention this?
# https://github.com/jwyang/faster-rcnn.pytorch/blob/f9d984d27b48a067b29792932bcb5321a39c1f09/lib/model/faster_rcnn/resnet.py

# Another good repo: https://github.com/potterhsu/easy-faster-rcnn.pytorch/blob/2c30c6d4ea57402c813294a499181b6ad710f858/model.py#L87
#
# Backbone needs to be split into feature extractor and classifier
# For ResNet, feature extractor is up through layer 3, and output will be (batch, 1024, H, W).
# This can be fed to RoI pool and will end up producing (N, 1024, 7, 7)
# This in turn can be fed to the classifier layer, which for ResNet is just layer 4 and will output (N, 2048, 4, 4).
# The last two dimensions must be eliminated by averaging: .mean(3).mean(2)
# This produces (N, 2048) which can be fed into detector classifier and regressor layers (inputs to this must be adjusted from 4096->2048)


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

from enum import Enum
from math import ceil
import torch as t
from torch import nn
from torch.nn import functional as F
import torchvision

from ..datasets import image
from .backbone import Backbone


class Architecture(Enum):
  ResNet50 = "ResNet50"
  ResNet101 = "ResNet101"
  ResNet152 = "ResNet152"


class FeatureExtractor(nn.Module):
  def __init__(self, resnet):
    super().__init__()

    # Feature extractor layers
    self._feature_extractor = nn.Sequential(
      resnet.conv1,     # 0
      resnet.bn1,       # 1
      resnet.relu,      # 2
      resnet.maxpool,   # 3
      resnet.layer1,    # 4
      resnet.layer2,    # 5
      resnet.layer3     # 6
    )

    # Freeze initial layers
    self._freeze(resnet.conv1)
    self._freeze(resnet.bn1)
    self._freeze(resnet.layer1)

    # Ensure that all batchnorm layers are frozen
    self._freeze_batchnorm(self._feature_extractor)

  def train(self, mode=True):
    super().train(mode)
    if mode:
      # Set fixed blocks to be in eval mode
      self._feature_extractor.eval()
      self._feature_extractor[5].train()
      self._feature_extractor[6].train()

      def set_bn_eval(module):
        if type(module) == nn.BatchNorm2d:
          module.eval()

      self._feature_extractor.apply(set_bn_eval)

  def forward(self, image_data):
    y = self._feature_extractor(image_data)
    return y

  @staticmethod
  def _freeze(layer):
    for name, parameter in layer.named_parameters():
      parameter.requires_grad = False

  def _freeze_batchnorm(self, block):
    for child in block.modules():
      if type(child) == nn.BatchNorm2d:
        self._freeze(layer = child)


class PoolToFeatureVector(nn.Module):
  def __init__(self, resnet):
    super().__init__()
    self._layer4 = resnet.layer4
    self._freeze_batchnorm(self._layer4)

  def train(self, mode=True):
    super().train(mode)
    if mode:
      def set_bn_eval(module):
        if type(module) == nn.BatchNorm2d:
          module.eval()
      self._layer4.apply(set_bn_eval)

  def forward(self, rois):
    y = self._layer4(rois)  # (N, 1024, 7, 7) -> (N, 2048, 4, 4)
    y = y.mean(-1).mean(-1) # use mean to remove last two dimensions -> (N, 2048)
    return y

  @staticmethod
  def _freeze(layer):
    for name, parameter in layer.named_parameters():
      parameter.requires_grad = False

  def _freeze_batchnorm(self, block):
    for child in block.modules():
      if type(child) == nn.BatchNorm2d:
        self._freeze(layer = child)


class ResNetBackbone(Backbone):
  def __init__(self, architecture):
    super().__init__()

    # Backbone properties. Image preprocessing parameters are common to all
    # Torchvision ResNet models and are described in the documentation, e.g.,
    # https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html#torchvision.models.resnet50
    self.feature_map_channels = 1024  # feature extractor output channels
    self.feature_pixels = 16          # ResNet feature maps are 1/16th of the original image size, similar to VGG-16 feature extractor
    self.feature_vector_size = 2048   # linear feature vector size after pooling
    self.image_preprocessing_params = image.PreprocessingParams(channel_order = image.ChannelOrder.RGB, scaling = 1.0 / 255.0, means = [ 0.485, 0.456, 0.406 ], stds = [ 0.229, 0.224, 0.225 ])

    # Construct model and pre-load with ImageNet weights
    if architecture == Architecture.ResNet50:
      resnet = torchvision.models.resnet50(weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    elif architecture == Architecture.ResNet101:
      resnet = torchvision.models.resnet101(weights = torchvision.models.ResNet101_Weights.IMAGENET1K_V1)
    elif architecture == Architecture.ResNet152:
      resnet = torchvision.models.resnet152(weights = torchvision.models.ResNet152_Weights.IMAGENET1K_V1)
    else:
      raise ValueError("Invalid ResNet architecture value: %s" % str(architecture))
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