import torch as t
from torch import nn
from torch.nn import functional as F
from torchvision.ops import RoIPool


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

    # Load Caffe model
    #TODO: these initial conditions may differ from the Chen model. 
    state = t.load("vgg16_caffe.pth")
    self._fc1.weight.data = state["classifier.0.weight"]
    self._fc1.bias.data = state["classifier.0.bias"]
    self._fc2.weight.data = state["classifier.3.weight"]
    self._fc2.bias.data = state["classifier.3.bias"]

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
