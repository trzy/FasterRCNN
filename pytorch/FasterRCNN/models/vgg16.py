#
# Faster R-CNN in PyTorch and TensorFlow 2 w/ Keras
# pytorch/FasterRCNN/models/vgg16.py
# Copyright 2021-2022 Bart Trzynadlowski
#
# PyTorch implementation of the VGG-16 backbone for use as a feature extractor
# in Faster R-CNN. Only the convolutional layers are used.
#

import torch as t
from torch import nn
from torch.nn import functional as F
import torchvision

from .feature_extractor import Backbone

#
# Custom VGG-16 Backbone
#

class CustomVGG16FeatureExtractor(nn.Module):
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


class CustomVGG16PoolToFeatureVector(nn.Module):
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


# Custom VGG-16 implementation, following the original paper
class CustomVGG16Backbone(Backbone):
  def __init__(self, dropout_probability):
    super().__init__()

    # Backbone properties
    self.feature_map_channels = 512
    self.feature_pixels = 16
    self.feature_vector_size = 4096

    # Feature extractor: convert input image (batch_size, channels, height, width)
    # to a feature map of shape (batch_size, 512, height // 16, width // 16)
    self.feature_extractor = CustomVGG16FeatureExtractor()

    # Conversion of pooled features to head input
    self.pool_to_feature_vector = CustomVGG16PoolToFeatureVector(dropout_probability = dropout_probability)

  def compute_feature_map_shape(self, image_shape):
    image_width = image_shape[-1]
    image_height = image_shape[-2]
    return (self.out_channels, image_height // self.feature_pixels, image_width // self.feature_pixels)


#
# Torchvision VGG-16-based Backbone
#

class TorchVGG16FeatureExtractor(nn.Module):
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


class TorchVGG16PoolToFeatureVector(nn.Module):
  def __init__(self, vgg16):
    super().__init__()

    # Get classifier layers
    assert len(vgg16.classifier) == 7 and type(vgg16.classifier[-1]) == nn.modules.linear.Linear, "Torchvision VGG-16 model does not have expected architecture"
    self._layers = vgg16.classifier[0:-1] # get all classifier layers except for the final one (4096 -> 1000 ImageNet class output)

  def forward(self, rois):
    rois = rois.reshape((rois.shape[0], 512 * 7 * 7)) # flatten each RoI: (N, 512*7*7)
    return self._layers(rois)


# Torchvision VGG-16 model with built-in pre-trained weights
class TorchVGG16Backbone(Backbone):
  def __init__(self, dropout_probability):
    super().__init__()

    # Backbone properties
    self.feature_map_channels = 512
    self.feature_pixels = 16
    self.feature_vector_size = 4096

    # Construct model with given dropout probability and pre-loaded with ImageNet weights
    vgg16 = torchvision.models.vgg16(weights = "IMAGENET1K_V1", dropout = dropout_probability)
    print("Loaded IMAGENET1K_V1 pre-trained weights for Torchvision VGG-16 feature extractor")

    # Feature extractor: convert input image (batch_size, channels, height, width)
    # to a feature map of shape (batch_size, 512, height // 16, width // 16)
    self.feature_extractor = TorchVGG16FeatureExtractor(vgg16 = vgg16)

    # Conversion of pooled features to head input
    self.pool_to_feature_vector = TorchVGG16PoolToFeatureVector(vgg16 = vgg16)

  def compute_feature_map_shape(self, image_shape):
    image_width = image_shape[-1]
    image_height = image_shape[-2]
    return (self.feature_map_channels, image_height // self.feature_pixels, image_width // self.feature_pixels)