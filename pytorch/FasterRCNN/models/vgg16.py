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
from torchvision.models import vgg16


class FeatureExtractor(nn.Module):
  def __init__(self):
    super().__init__()

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
