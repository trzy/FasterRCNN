#
# Faster R-CNN in PyTorch and TensorFlow 2 w/ Keras
# pytorch/FasterRCNN/models/vgg16.py
# Copyright 2021-2022 Bart Trzynadlowski
#
# PyTorch implementation of the VGG-16 backbone for use as a feature extractor
# in Faster R-CNN. Only the convolutional layers are used. This implementation
# is fully custom, in contrast to using Torchvision's pre-trained layers. Its
# weights are incompatible with Torchvision's but *are* compatible with the
# Caffe implementation of VGG-16.
#

import torch as t
from torch import nn
from torch.nn import functional as F
import torchvision

from ..datasets import image
from .backbone import Backbone


class FeatureExtractor(nn.Module):
  def __init__(self):
    super().__init__()

    # Define network
    self._block1_conv1 = nn.Conv2d(in_channels = 3,  out_channels = 64, kernel_size = (3, 3), stride = 1, padding = "same")
    self._block1_conv2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3, 3), stride = 1, padding = "same")
    self._block1_pool = nn.MaxPool2d(kernel_size = (2, 2), stride = 2)

    self._block2_conv1 = nn.Conv2d(in_channels = 64,  out_channels = 128, kernel_size = (3, 3), stride = 1, padding = "same")
    self._block2_conv2 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = (3, 3), stride = 1, padding = "same")
    self._block2_pool = nn.MaxPool2d(kernel_size = (2, 2), stride = 2)

    self._block3_conv1 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = (3, 3), stride = 1, padding = "same")
    self._block3_conv2 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (3, 3), stride = 1, padding = "same")
    self._block3_conv3 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (3, 3), stride = 1, padding = "same")
    self._block3_pool = nn.MaxPool2d(kernel_size = (2, 2), stride = 2)

    self._block4_conv1 = nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = (3, 3), stride = 1, padding = "same")
    self._block4_conv2 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = (3, 3), stride = 1, padding = "same")
    self._block4_conv3 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = (3, 3), stride = 1, padding = "same")
    self._block4_pool = nn.MaxPool2d(kernel_size = (2, 2), stride = 2)

    self._block5_conv1 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = (3, 3), stride = 1, padding = "same")
    self._block5_conv2 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = (3, 3), stride = 1, padding = "same")
    self._block5_conv3 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = (3, 3), stride = 1, padding = "same")

    # Freeze first two convolutional blocks
    self._block1_conv1.weight.requires_grad = False
    self._block1_conv1.bias.requires_grad = False
    self._block1_conv2.weight.requires_grad = False
    self._block1_conv2.bias.requires_grad = False

    self._block2_conv1.weight.requires_grad = False
    self._block2_conv1.bias.requires_grad = False
    self._block2_conv2.weight.requires_grad = False
    self._block2_conv2.bias.requires_grad = False

  def forward(self, image_data):
    """
    Converts input images into feature maps using VGG-16 convolutional layers.

    Parameters
    ----------
    image_data : torch.Tensor
      A tensor of shape (batch_size, channels, height, width) representing
      images normalized using the VGG-16 convention (BGR, ImageNet channel-wise
      mean-centered).

    Returns
    -------
    torch.Tensor
      Feature map of shape (batch_size, 512, height // 16, width // 16).
    """
    y = F.relu(self._block1_conv1(image_data))
    y = F.relu(self._block1_conv2(y))
    y = self._block1_pool(y)

    y = F.relu(self._block2_conv1(y))
    y = F.relu(self._block2_conv2(y))
    y = self._block2_pool(y)

    y = F.relu(self._block3_conv1(y))
    y = F.relu(self._block3_conv2(y))
    y = F.relu(self._block3_conv3(y))
    y = self._block3_pool(y)

    y = F.relu(self._block4_conv1(y))
    y = F.relu(self._block4_conv2(y))
    y = F.relu(self._block4_conv3(y))
    y = self._block4_pool(y)

    y = F.relu(self._block5_conv1(y))
    y = F.relu(self._block5_conv2(y))
    y = F.relu(self._block5_conv3(y))

    return y


class PoolToFeatureVector(nn.Module):
  def __init__(self, dropout_probability):
    super().__init__()

    # Define network layers
    self._fc1 = nn.Linear(in_features = 512 * 7 * 7, out_features = 4096)
    self._fc2 = nn.Linear(in_features = 4096, out_features = 4096)

    # Dropout layers
    self._dropout1 = nn.Dropout(p = dropout_probability)
    self._dropout2 = nn.Dropout(p = dropout_probability)

  def forward(self, rois):
    """
    Converts RoI-pooled features into a linear feature vector suitable for use
    with the detector heads (classifier and regressor).

    Parameters
    ----------
    rois : torch.Tensor
      Output of RoI pool layer, of shape (N, 512, 7, 7).

    Returns
    -------
    torch.Tensor
      Feature vector of shape (N, 4096).
    """

    rois = rois.reshape((rois.shape[0], 512*7*7))  # flatten each RoI: (N, 512*7*7)
    y1o = F.relu(self._fc1(rois))
    y1 = self._dropout1(y1o)
    y2o = F.relu(self._fc2(y1))
    y2 = self._dropout2(y2o)

    return y2


class VGG16Backbone(Backbone):
  def __init__(self, dropout_probability):
    super().__init__()

    # Backbone properties
    self.feature_map_channels = 512
    self.feature_pixels = 16
    self.feature_vector_size = 4096
    self.image_preprocessing_params = image.PreprocessingParams(channel_order = image.ChannelOrder.BGR, scaling = 1.0, means = [ 103.939, 116.779, 123.680 ], stds = [ 1, 1, 1 ])

    # Feature extractor: convert input image (batch_size, channels, height, width)
    # to a feature map of shape (batch_size, 512, height // 16, width // 16)
    self.feature_extractor = FeatureExtractor()

    # Conversion of pooled features to head input
    self.pool_to_feature_vector = PoolToFeatureVector(dropout_probability = dropout_probability)

  def compute_feature_map_shape(self, image_shape):
    image_width = image_shape[-1]
    image_height = image_shape[-2]
    return (self.feature_map_channels, image_height // self.feature_pixels, image_width // self.feature_pixels)