import torch as t
from torch import nn
from torch.nn import functional as F
from torchvision.ops import RoIPool
from torchvision.models import vgg16

from FasterRCNN import utils


class DetectorNetwork(nn.Module):
  def __init__(self, num_classes):
    super().__init__()
  
    # Define network
    self._roi_pool = RoIPool(output_size = (7, 7), spatial_scale = 1.0 / 16.0)
    self._fc1 = nn.Linear(in_features = 512*7*7, out_features = 4096)
    self._fc2 = nn.Linear(in_features = 4096, out_features = 4096)
    self._classifier = nn.Linear(in_features = 4096, out_features = num_classes)
    self._regressor = nn.Linear(in_features = 4096, out_features = (num_classes - 0) * 4) 
   
    # Initialize weights
    self._classifier.weight.data.normal_(mean = 0.0, std = 0.01)
    self._classifier.bias.data.zero_()
    self._regressor.weight.data.normal_(mean = 0.0, std = 0.001)
    self._regressor.bias.data.zero_()

    # Load pre-trained weights from PyTorch VGG-16 implementation
    self._load_pretrained_vgg16_weights()

  def forward(self, feature_map, proposals):
    # Batch size of one for now, so no need to associate proposals with batches
    assert feature_map.shape[0] == 1, "Batch size must be 1"
    batch_idxs = t.zeros((proposals.shape[0], 1)).cuda()

    # (N, 5) tensor of (batch_idx, x1, y1, x2, y2)
    proposals = t.from_numpy(proposals).cuda()
    indexed_proposals = t.cat([ batch_idxs, proposals ], dim = 1)
    indexed_proposals = indexed_proposals[:, [ 0, 2, 1, 4, 3 ]] # each row, (batch_idx, y1, x1, y2, x2) -> (batch_idx, x1, y1, x2, y2)

    # RoI pooling: (N, 512, 7, 7)
    rois = self._roi_pool(feature_map, indexed_proposals)
    rois = rois.reshape((rois.shape[0], 512*7*7)) # flatten each RoI: (N, 512*7*7)

    # Forward propagate
    y1 = F.relu(self._fc1(rois))
    y2 = F.relu(self._fc2(y1))
    classes_raw = self._classifier(y2)
    classes = F.softmax(classes_raw, dim = 1)
    regressions = self._regressor(y2)

    return classes, regressions

  def load_caffe_vgg16_weights(self, state):
    # Load Caffe weights, which were trained on a model that used the same
    # image normalization as we do (unlike the torchvision version)
    self._fc1.weight.data = state["classifier.0.weight"]
    self._fc1.bias.data = state["classifier.0.bias"]
    self._fc2.weight.data = state["classifier.3.weight"]
    self._fc2.bias.data = state["classifier.3.bias"]

  def load_keras_vgg16_weights(self, hdf5_file):
    weights_fc1 = utils.load_keras_weights(hdf5_file = hdf5_file, layer_name = "fc1")
    if weights_fc1 is not None:
      #
      # The fc1 layer in Keras takes as input a flattened (7, 7, 512) map from
      # the RoI pool layer. Here in PyTorch, it is (512, 7, 7). Keras stores
      # weights as (25088, 4096), which is equivalent to (7, 7, 512, 4096), as
      # per Keras channels-last convention. To convert to PyTorch, we must
      # first transpose to (512, 7, 7, 4096), then flatten to (25088, 4096),
      # and, lastly, transpose to (4096, 25088).
      #
      weights_fc1 = weights_fc1.reshape((7, 7, 512, 4096))
      weights_fc1 = weights_fc1.transpose((2, 0, 1, 3)) # (512, 7, 7, 4096)
      weights_fc1 = weights_fc1.reshape((-1, 4096))     # (25088, 4096)
      weights_fc1 = weights_fc1.transpose([ 1, 0 ])     # (4096, 25088)
      self._fc1.weight.data = t.from_numpy(weights_fc1).to("cuda")
    utils.set_keras_biases(layer = self._fc1, hdf5_file = hdf5_file, layer_name = "fc1")
    
    weights_fc2 = utils.load_keras_weights(hdf5_file = hdf5_file, layer_name = "fc2")
    if weights_fc2 is not None:
      #
      # Due to the adjustment for fc1, fc2 should be possible to load with only
      # a tranpose of the two components (in_dimension, out_dimension) ->
      # (out_dimension, in_dimension).
      #
      weights_fc2 = weights_fc2.transpose([ 1, 0 ])
      self._fc2.weight.data = t.from_numpy(weights_fc2).to("cuda")
    utils.set_keras_biases(layer = self._fc2, hdf5_file = hdf5_file, layer_name = "fc2")

  def _load_pretrained_vgg16_weights(self):
    # Get the pre-trained torchvision model with weights based on VGG-16-style
    # image normalization (mean subtraction, BGR order)
    vgg16_model = vgg16(pretrained = False)
    torchvision_model_layers = list(vgg16_model.classifier)

    layer_to_torchvision_model_index = {
      self._fc1: 0,
      self._fc2: 3
    }

    for layer, index in layer_to_torchvision_model_index.items():
      layer.weight.data = torchvision_model_layers[index].weight.data.float()
      layer.bias.data = torchvision_model_layers[index].bias.data.float() 


def class_loss(predicted_classes, y_true):
  """
  Computes detector class loss. 

  Parameters
  ----------
  predicted_classes : torch.Tensor
    RoI predicted classes as categorical vectors, (N, num_classes).
  y_true : torch.Tensor
    RoI class labels as categorical vectors, (N, num_classes).

  Returns
  -------
  torch.Tensor
    Scalar loss.
  """
  epsilon = 1e-7
  scale_factor = 1.0
  cross_entropy_per_row = -(y_true * t.log(predicted_classes + epsilon)).sum(dim = 1)
  N = cross_entropy_per_row.shape[0] + epsilon
  cross_entropy = t.sum(cross_entropy_per_row) / N
  return scale_factor * cross_entropy

def regression_loss(predicted_regressions, y_true):
  epsilon = 1e-7
  scale_factor = 1.0
  sigma = 1.0
  sigma_squared = sigma * sigma

  # We want to unpack the regression targets and the mask of valid targets into
  # tensors each of the same shape as the predicted: 
  #   (num_proposals, 4*(num_classes-1))
  # y_true has shape:
  #   (num_proposals, 2, 4*(num_classes-1))
  y_mask = y_true[:,0,:]
  y_true_targets = y_true[:,1,:]

  # Compute element-wise loss using robust L1 function for all 4 regression
  # targets
  x = y_true_targets - predicted_regressions
  x_abs = t.abs(x)
  is_negative_branch = (x < (1.0 / sigma_squared)).float()
  R_negative_branch = 0.5 * x * x * sigma_squared
  R_positive_branch = x_abs - 0.5 / sigma_squared
  losses = is_negative_branch * R_negative_branch + (1.0 - is_negative_branch) * R_positive_branch

  # Normalize to number of proposals (e.g., 128). Although this may not be
  # what the paper does, it seems to work. Other implemetnations do this.
  # Using e.g., the number of positive proposals will cause the loss to 
  # behave erratically because sometimes N will become very small.  
  N = y_true.shape[0] + epsilon
  relevant_loss_terms = y_mask * losses
  return scale_factor * t.sum(relevant_loss_terms) / N
