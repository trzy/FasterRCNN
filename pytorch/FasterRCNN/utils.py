import h5py
import numpy as np
import torch as t


def no_grad(func):
  def wrapper_nograd(*args, **kwargs):
    with t.no_grad():
      return func(*args, **kwargs)
  return wrapper_nograd

def load_keras_weights(hdf5_file, layer_name):
  """
  Loads Keras-formatted weights from an HDF5 file and returns them.

  Parameters
  ----------
  hdf5_file : h5py.File
    Opened HDF5 file object.
  layer_name : str
    Name of layer to load. E.g., "fc1".

  Returns
  -------
  np.ndarray
    Weights or None if layer not found.
  """
  primary_keypath = "model_weights/" + layer_name
  for keypath, node in hdf5_file[primary_keypath].items():
    if keypath.startswith("conv") or keypath.startswith("dense"):
      kernel_keypath = "/".join([primary_keypath, keypath, "kernel:0"])
      return np.array(hdf5_file[kernel_keypath]).astype(np.float32)
  return None

def set_keras_conv2d_weights(layer, hdf5_file, layer_name, keras_shape = None):
  """
  Loads Keras-formatted 2D convolutional kernel weights from an HDF5 file and
  into a PyTorch layer. Keras stores kernels as:

    (kernel_height, kernel_width, channels_in, channels_out)
  
  PyTorch:

    (channels_out, channels_in, kernel_height, kernel_width)

  Parameters
  ----------
  layer : torch.nn.Module
    Layer on which to set weights, if they exist in the file.
  hdf5_file : h5py.File
    Opened HDF5 file object.
  layer_name : str
    Name of layer to load. E.g., "block1_conv1".
  keras_shape : tuple
    Original Keras shape. If specified, weights are reshaped to this shape
    before being transposed to PyTorch format.
  """
  w = load_keras_weights(hdf5_file = hdf5_file, layer_name = layer_name)
  if w is not None:
    if keras_shape is not None:
      w = w.reshape(keras_shape)
    w = w.transpose([ 3, 2, 0, 1 ])
    layer.weight.data = t.from_numpy(w).to("cuda")

def set_keras_biases(layer, hdf5_file, layer_name):
  """
  Loads biases from an HDF5 file into a PyTorch layer.

  Parameters
  ----------
  layer : torch.nn.Module
    Layer on which to set weights, if they exist in the file.
  hdf5_file : h5py.File
    Opened HDF5 file object.
  layer_name : str
    Name of the layer to load. E.g., "block1_conv1".

  Returns
  -------
  torch.Tensor
    Bias vector.
  """
  primary_keypath = "model_weights/" + layer_name
  for keypath, node in hdf5_file[primary_keypath].items():
    if keypath.startswith("conv") or keypath.startswith("dense"):
      bias_keypath = "/".join([primary_keypath, keypath, "bias:0"])
      b = np.array(hdf5_file[bias_keypath]).astype(np.float32)
      layer.bias.data = t.from_numpy(b).to("cuda")

