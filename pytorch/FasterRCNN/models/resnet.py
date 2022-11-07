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

import math
from math import ceil
import torch as t
from torch import nn
from torch.nn import functional as F
import torchvision

from .feature_extractor import Backbone




def conv3x3(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
           padding=1, bias=False)


class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(BasicBlock, self).__init__()
    self.conv1 = conv3x3(inplanes, planes, stride)
    self.bn1 = nn.BatchNorm2d(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = nn.BatchNorm2d(planes)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class Bottleneck(nn.Module):
  expansion = 4

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(Bottleneck, self).__init__()
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False) # change
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, # change
                 padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(planes * 4)
    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class ResNet(nn.Module):
  def __init__(self, block, layers, num_classes=1000):
    self.inplanes = 64
    super(ResNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                 bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False) # change
    self.layer1 = self._make_layer(block, 64, layers[0])
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
    self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
    # it is slightly better whereas slower to set stride = 1
    # self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
    self.avgpool = nn.AvgPool2d(7)
    self.fc = nn.Linear(512 * block.expansion, num_classes)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * block.expansion,
              kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * block.expansion),
      )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)

    return x

























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
    #y = F.adaptive_max_pool2d(y, output_size = 1).squeeze()
    return y

  @staticmethod
  def _freeze(layer):
    for name, parameter in layer.named_parameters():
      parameter.requires_grad = False

  def _freeze_batchnorm(self, block):
    for child in block.modules():
      if type(child) == nn.BatchNorm2d:
        self._freeze(layer = child)


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


    # Freeze initial layers
    self._freeze(self._feature_extractor[0])
    self._freeze(self._feature_extractor[1])
    self._freeze(self._feature_extractor[4])
    #self._freeze(self._feature_extractor[5])

    #self._freeze(resnet.conv1)
    #self._freeze(resnet.bn1)
    #self._freeze(resnet.layer1)
    #self._freeze(resnet.layer2)

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
      #self.RCNN_top.apply(set_bn_eval)


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
    state = resnet.state_dict()

    resnet = ResNet(Bottleneck, [ 3, 4, 6, 3])
    resnet.load_state_dict(state)

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