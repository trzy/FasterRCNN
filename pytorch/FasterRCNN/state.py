#
# Faster R-CNN in PyTorch and TensorFlow 2 w/ Keras
# python/pytorch/FasterRCNN/state.py
# Copyright 2021-2022 Bart Trzynadlowski
#
# Model state file management code. Loading and saving of weights.
#

import h5py
import numpy as np
import torch as t


def _load_keras_weights(hdf5_file, layer_name):
  """
  Loads Keras-formatted weights from an HDF5 file and returns them as a PyTorch
  tensor.

  Parameters
  ----------
  hdf5_file : h5py.File
    Opened HDF5 file object.
  layer_name : str
    Name of layer to load. E.g., "fc1".

  Returns
  -------
  torch.Tensor
    Weights or None if layer not found.
  """
  primary_keypath = "model_weights/" + layer_name
  for keypath, node in hdf5_file[primary_keypath].items():
    if keypath.startswith("conv") or keypath.startswith("dense"):
      kernel_keypath = "/".join([primary_keypath, keypath, "kernel:0"])
      weights = np.array(hdf5_file[kernel_keypath]).astype(np.float32)
      return t.from_numpy(weights).cuda()
  return None

def _load_keras_biases(hdf5_file, layer_name):
  """
  Loads Keras-formatted biases from an HDF5 file and returns them as a PyTorch
  vector.

  Parameters
  ----------
  hdf5_file : h5py.File
    Opened HDF5 file object.
  layer_name : str
    Name of the layer to load. E.g., "block1_conv1".

  Returns
  -------
  torch.Tensor
    Bias vector or None if layer not be found.
  """
  primary_keypath = "model_weights/" + layer_name
  for keypath, node in hdf5_file[primary_keypath].items():
    if keypath.startswith("conv") or keypath.startswith("dense"):
      bias_keypath = "/".join([primary_keypath, keypath, "bias:0"])
      biases = np.array(hdf5_file[bias_keypath]).astype(np.float32)
      return t.from_numpy(biases).cuda()
  return None

def _load_keras_layer(hdf5_file, layer_name):
  """
  Loads Keras-formatted weights and biases from an HDF5 file and returns them
  as PyTorch tensors.

  Parameters
  ----------
  hdf5_file : h5py.File
    Opened HDF5 file object.
  layer_name : str
    Name of layer to load. E.g., "fc1".

  Returns
  -------
  torch.Tensor, torch.Tensor
    Weights and biases. One or both can be None if not found.
  """
  return _load_keras_weights(hdf5_file = hdf5_file, layer_name = layer_name), _load_keras_biases(hdf5_file = hdf5_file, layer_name = layer_name)

def _load_keras_conv2d_layer(hdf5_file, layer_name, keras_shape = None):
  """
  Loads Keras-formatted 2D convolutional kernel weights and biases from an HDF5
  file and returns them as PyTorch tensors. Keras stores kernels as:

    (kernel_height, kernel_width, channels_in, channels_out)
  
  PyTorch:

    (channels_out, channels_in, kernel_height, kernel_width)

  Parameters
  ----------
  hdf5_file : h5py.File
    Opened HDF5 file object.
  layer_name : str
    Name of layer to load. E.g., "block1_conv1".
  keras_shape : tuple
    Original Keras shape. If specified, weights are reshaped to this shape
    before being transposed to PyTorch format.

  Returns
  -------
  torch.Tensor, torch.Tensor
    Weights and biases. One or both can be None if not found.
  """
  weights, biases = _load_keras_layer(hdf5_file = hdf5_file, layer_name = layer_name)
  if weights is not None and biases is not None:
    if keras_shape is not None:
      weights = weights.reshape(keras_shape)
    weights = weights.permute([ 3, 2, 0, 1 ])
  return weights, biases

def _load_vgg16_from_bart_keras_model(filepath):
  missing_layers = []
  state = {}
  file = h5py.File(filepath, "r")

  # Feature extractor
  keras_layers = [
    "block1_conv1",
    "block1_conv2",
    "block2_conv1",
    "block2_conv2",
    "block3_conv1",
    "block3_conv2",
    "block3_conv3",
    "block4_conv1",
    "block4_conv2",
    "block4_conv3",
    "block5_conv1",
    "block5_conv2",
    "block5_conv3"
  ]
  for layer_name in keras_layers:
    weights, biases = _load_keras_conv2d_layer(hdf5_file = file, layer_name = layer_name)
    if weights is not None and biases is not None:
      state["_stage1_feature_extractor._" + layer_name + ".weight"] = weights 
      state["_stage1_feature_extractor._" + layer_name + ".bias"] = biases
    else:
      missing_layers.append(layer_name)

  # Detector
  weights, biases = _load_keras_layer(hdf5_file = file, layer_name = "fc1")
  if weights is not None and biases is not None:
    # The fc1 layer in Keras takes as input a flattened (7, 7, 512) map from
    # the RoI pool layer. Here in PyTorch, it is (512, 7, 7). Keras stores
    # weights as (25088, 4096), which is equivalent to (7, 7, 512, 4096), as
    # per Keras channels-last convention. To convert to PyTorch, we must
    # first transpose to (512, 7, 7, 4096), then flatten to (25088, 4096),
    # and, lastly, transpose to (4096, 25088).
    weights = weights.reshape((7, 7, 512, 4096))
    weights = weights.permute([ 2, 0, 1, 3 ]) # (512, 7, 7, 4096)
    weights = weights.reshape((-1, 4096))     # (25088, 4096)
    weights = weights.permute([ 1, 0 ])       # (4096, 25088)
    state["_stage3_detector_network._fc1.weight"] = weights
    state["_stage3_detector_network._fc1.bias"] = biases
  else:
    missing_layers.append("fc1") 
  weights, biases = _load_keras_layer(hdf5_file = file, layer_name = "fc2")
  if weights is not None and biases is not None:
    # Due to the adjustment for fc1, fc2 can be loaded with only a transpose
    # of the two components (in_dimension, out_dimension) -> 
    # (out_dimension, in_dimension).
    state["_stage3_detector_network._fc2.weight"] = weights.permute([ 1, 0 ])
    state["_stage3_detector_network._fc2.bias"] = biases
  else:
    missing_layers.append("fc2")

  # Anything missing?
  if len(missing_layers) > 0:    
    print("Some layers were missing from '%s' and not loaded: %s" % (filepath, ", ".join(missing_layers)))

  return state

def _load_vgg16_from_caffe_model(filepath):
  state = {}
  caffe = t.load(filepath)

  # Attempt to load all layers
  mapping = {
    "features.0.":    "_stage1_feature_extractor._block1_conv1",
    "features.2.":    "_stage1_feature_extractor._block1_conv2",
    "features.5.":    "_stage1_feature_extractor._block2_conv1",
    "features.7.":    "_stage1_feature_extractor._block2_conv2",
    "features.10.":   "_stage1_feature_extractor._block3_conv1",
    "features.12.":   "_stage1_feature_extractor._block3_conv2",
    "features.14.":   "_stage1_feature_extractor._block3_conv3",
    "features.17.":   "_stage1_feature_extractor._block4_conv1",
    "features.19.":   "_stage1_feature_extractor._block4_conv2",
    "features.21.":   "_stage1_feature_extractor._block4_conv3",
    "features.24.":   "_stage1_feature_extractor._block5_conv1",
    "features.26.":   "_stage1_feature_extractor._block5_conv2",
    "features.28.":   "_stage1_feature_extractor._block5_conv3",
    "classifier.0.":  "_stage3_detector_network._fc1",
    "classifier.3.":  "_stage3_detector_network._fc2"
  }
  missing_layers = set([ layer_name[:-1] for layer_name in mapping.keys() ])  # we will remove as we load
  for key, tensor in caffe.items():
    caffe_layer_name = ".".join(key.split(".")[0:2])  # grab first two parts
    caffe_key = caffe_layer_name + "."                # add trailing '.' for key in mapping dict
    if caffe_key in mapping:
      weight_key = caffe_key + "weight"
      bias_key = caffe_key + "bias"
      if weight_key in caffe and bias_key in caffe:
        state[mapping[caffe_key] + ".weight"] = caffe[weight_key]
        state[mapping[caffe_key] + ".bias"] = caffe[bias_key]
        missing_layers.discard(caffe_layer_name)

  # If *all* were missing, this file must not contain the Caffe VGG-16 model
  if len(missing_layers) == len(mapping):
    raise ValueError("File '%s' is not a Caffe VGG-16 model" % filepath)

  if len(missing_layers) > 0:
    print("Some layers were missing from '%s' and not loaded: %s" % (filepath, ", ".join(missing_layers)))
    
  return state

def load(model, filepath):
  """
  Load model wieghts and biases from a file. We support 3 different formats:
  
    - PyTorch state files containing our complete model as-is
    - PyTorch state files containing only VGG-16 layers trained in Caffe (i.e.,
      the published reference implementation of VGG-16). These are compatible
      with the VGG-16 image normalization used here, unlike the torchvision
      VGG-16 implementation. The Caffe state file can be found online and is
      usually named vgg16_caffe.pth.
    - Keras h5 state file containing only VGG-16 layers trained by my own
      VGG-16 model (github.com/trzy/VGG16).

  Parameters
  ----------
  model : torch.nn.Module
    The complete Faster R-CNN model to load weights and biases into.
  filepath : str
    File to load.
  """

  state = None

  # Keras?
  try:
    state = _load_vgg16_from_bart_keras_model(filepath = filepath)
    print("Loaded initial VGG-16 layer weights from Keras model '%s'" % filepath)
  except:
    pass

  # Caffe?
  if state is None:
    try:
      state = _load_vgg16_from_caffe_model(filepath = filepath)
      print("Loaded initial VGG-16 layer weights from Caffe model '%s'" % filepath)
    except Exception as e:
      pass

  # Assume complete PyTorch state
  if state is None:
    state = t.load(filepath)
    if "model_state_dict" not in state:
      raise KeyError("Model state file '%s' is missing top-level key 'model_state_dict'" % filepath)
    state = state["model_state_dict"]

  # Load
  try:
    model.load_state_dict(state)
    print("Loaded initial weights from '%s'" % filepath)
  except Exception as e:
    print(e)
    return

class BestWeightsTracker:
  def __init__(self, filepath):
    self._filepath = filepath
    self._best_state = None
    self._best_mAP = 0

  def on_epoch_end(self, model, epoch, mAP):
    if mAP > self._best_mAP:
      self._best_mAP = mAP
      self._best_state = { "epoch": epoch, "model_state_dict": model.state_dict() }

  def save_best_weights(self, model):
    if self._best_state is not None:
      t.save(self._best_state, self._filepath)
      print("Saved best model weights (Mean Average Precision = %1.2f%%) to '%s'" % (self._best_mAP, self._filepath))
