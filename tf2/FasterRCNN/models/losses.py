#
#  """
#  TODO: convert to unit test
#  y_true = np.array([
#    [1, 0, 0, 0],
#    [1, 0, 0, 0],
#    [0, 0, 1, 0],
#    [0, 0, 0, 1]
#  ]).reshape((2,2,4))
#  y_predicted = np.array([
#    [1, 0, 0, 0],
#    [0, 1, 0, 0],
#    [1, 0, 0, 0],
#    [0, 0, 0, 1]
#  ]).reshape((2,2,4))
#
#  loss = K.eval(classifier_class_loss(y_true = K.variable(y_true), y_predicted = K.variable(y_predicted)))
#  print(loss)
#  print("---")
#  for j in range(2):
#    for i in range(2):
#      yt = K.variable(y_true[j,i])
#      yp = K.variable(y_predicted[j,i])
#      print(K.eval(K.categorical_crossentropy(yt, yp)))
#
#  exit()
#  
#  y_true = np.zeros((1,2,2,8))
#  y_true[0,0,0,:] = 1, 1, 1, 1, 0, 0, 0, 0
#  y_true[0,0,1,:] = 1, 2, 3, 3, 2, 2, 2, 2
#  y_true[0,1,0,:] = 0, 0, 0, 0, 0, 0, 0, 0
#  y_true[0,1,1,:] = 0, 3, 0, 0, 3, 2, 1, 0
#  y_predicted = np.array([
#    [1,2,3,4,5,6,7,8],
#    [0,1,2,3,3,2,1,0]
#  ]).reshape((1,2,8))
#
#  loss = K.eval(classifier_regression_loss(y_true = K.variable(y_true), y_predicted = K.variable(y_predicted)))
#  print(loss)
#  exit()
#  """
#

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

def rpn_class_loss_np(y_true, y_predicted):
  """
  NumPy reference implementation of objectness class loss function for
  immediate execution.
  """
  y_predicted_class = y_predicted
  y_true_class = y_true[:,:,:,:,1].reshape(y_predicted.shape)
  y_mask = y_true[:,:,:,:,0].reshape(y_predicted.shape)
  epsilon = 1e-7
  N_cls = float(np.count_nonzero(y_mask)) + epsilon
  loss_all_anchors = -(y_true_class * np.log(y_predicted + epsilon) + (1.0 - y_true_class) * np.log(1.0 - y_predicted + epsilon)) # binary cross-entropy, element-wise
  relevant_loss_terms = y_mask * loss_all_anchors
  return np.sum(relevant_loss_terms) / N_cls

def rpn_class_loss(y_true, y_predicted):
  """
  Keras implementation of RPN objectness class loss function.
  """
  y_predicted_class = tf.convert_to_tensor(y_predicted)
  y_true_class = tf.cast(tf.reshape(y_true[:,:,:,:,1], shape = tf.shape(y_predicted)), dtype = y_predicted.dtype)

  # y_true[:,:,:,0] is 1.0 for anchors included in the mini-batch
  y_mask = tf.cast(tf.reshape(y_true[:,:,:,:,0], shape = tf.shape(y_predicted)), dtype = y_predicted.dtype)

  # Compute how many anchors are actually used in the mini-batch (e.g.,
  # typically 256)
  N_cls = tf.cast(tf.math.count_nonzero(y_mask), dtype = tf.float32) + K.epsilon()

  # Compute element-wise loss for all anchors
  loss_all_anchors = K.binary_crossentropy(y_true_class, y_predicted_class)

  # Zero out the ones which should not have been included
  relevant_loss_terms = y_mask * loss_all_anchors

  # Sum the total loss and normalize by the number of anchors used
  return K.sum(relevant_loss_terms) / N_cls

def rpn_regression_loss_np(y_true, y_predicted):
  """
  NumPy reference implementation of box regression loss.
  """
  scale_factor = 1.0
  sigma = 3.0
  sigma_squared = sigma * sigma
  y_predicted_regression = y_predicted
  y_true_regression = y_true[:,:,:,:,4:8].reshape(y_predicted.shape)
  mask_shape = (y_true.shape[0], y_true.shape[1], y_true.shape[2], y_true.shape[3])
  y_included = y_true[:,:,:,:,0].reshape(mask_shape)
  y_positive = y_true[:,:,:,:,1].reshape(mask_shape)        # object
  y_mask = y_included * y_positive
  y_mask = np.repeat(y_mask, repeats = 4, axis = 3)
  N_cls = float(np.count_nonzero(y_included)) + 1e-9
  print("N_cls=",N_cls)
  x = y_true_regression - y_predicted_regression
  x_abs = np.sqrt(x * x)  # K.abs/tf.abs crash (Windows only?)
  is_negative_branch = np.less(x_abs, 1.0 / sigma_squared).astype(np.float)
  R_negative_branch = 0.5 * sigma_squared * x * x
  R_positive_branch = x_abs - 0.5 / sigma_squared
  loss_all_anchors = is_negative_branch * R_negative_branch + (1.0 - is_negative_branch) * R_positive_branch
  relevant_loss_terms = y_mask * loss_all_anchors
  return scale_factor * np.sum(relevant_loss_terms) / N_cls

def rpn_regression_loss(y_true, y_predicted):
  #TODO: factor this out as an actual configurable parameter and make this function return a loss function
  scale_factor = 1.0  # hyper-parameter that controls magnitude of regression loss and is chosen to make regression term comparable to class term
  sigma = 3.0         # see: https://github.com/rbgirshick/py-faster-rcnn/issues/89
  sigma_squared = sigma * sigma

  y_predicted_regression = tf.convert_to_tensor(y_predicted)
  y_true_regression = tf.cast(tf.reshape(y_true[:,:,:,:,4:8], shape = tf.shape(y_predicted)), dtype = y_predicted.dtype)

  # Include only anchors that are used in the mini-batch and which correspond
  # to objects (positive samples)
  mask_shape = tf.slice(tf.shape(y_true), begin = [0], size = [4])
  y_included = tf.cast(tf.reshape(y_true[:,:,:,:,0], mask_shape), dtype = y_predicted.dtype)
  y_positive = tf.cast(tf.reshape(y_true[:,:,:,:,1], mask_shape), dtype = y_predicted.dtype)
  y_mask = y_included * y_positive

  # y_mask is of the wrong shape. We have one value per (y,x,k) position but in
  # fact need to have 4 values (one for each of the regression variables). For
  # example, y_predicted might be (1,37,50,36) and y_mask will be (1,37,50,9).
  # We need to repeat the last dimension 4 times.
  y_mask = tf.repeat(y_mask, repeats = 4, axis = 3)

  # The paper normalizes by dividing by a quantity called N_reg, which is equal
  # to the total number of anchors (~2400) and then multiplying by lambda=10.
  # This does not make sense to me because we are summing over a mini-batch at
  # most, so we use N_cls here. I might be misunderstanding what is going on
  # but 10/2400 = 1/240 which is pretty close to 1/256 and the paper mentions
  # that training is relatively insensitve to choice of normalization.
  N_cls = tf.cast(tf.math.count_nonzero(y_included), dtype = tf.float32) + K.epsilon()

  # Compute element-wise loss using robust L1 function for all 4 regression
  # components
  x = y_true_regression - y_predicted_regression
  x_abs = tf.math.abs(x) #tf.sqrt(x * x)  # K.abs/tf.abs crash (Windows only?)
  is_negative_branch = tf.cast(tf.less(x_abs, 1.0 / sigma_squared), dtype = tf.float32)
  R_negative_branch = 0.5 * x * x * sigma_squared
  R_positive_branch = x_abs - 0.5 / sigma_squared
  loss_all_anchors = is_negative_branch * R_negative_branch + (1.0 - is_negative_branch) * R_positive_branch

  # Zero out the ones which should not have been included
  relevant_loss_terms = y_mask * loss_all_anchors
  return scale_factor * K.sum(relevant_loss_terms) / N_cls

def classifier_class_loss(y_true, y_predicted):
  """
  Keras implementation of classifier network classification loss. The inputs
  are shaped (M,N,C), where M is the number of batches (i.e., 1), N is the
  number of proposed RoIs to classify, and C is the number of classes. One-hot
  encoding is used, hence categorical crossentropy.
  """
  scale_factor = 1.0
  return scale_factor * K.mean(K.categorical_crossentropy(y_true, y_predicted))

def classifier_regression_loss(y_true, y_predicted):
  scale_factor = 1.0
  sigma = 1.0
  sigma_squared = sigma * sigma

  # We want to unpack the regression targets and the mask of valid targets into
  # tensors each of the same shape as the predicted: 
  #   (batch_size, num_proposals, 4*(num_classes-1))
  # y_true has shape:
  #   (batch_size, num_proposals, 2, 4*(num_classes-1))
  y_mask = y_true[:,:,0,:]
  y_true_targets = y_true[:,:,1,:]

  # Compute element-wise loss using robust L1 function for all 4 regression
  # targets
  x = y_true_targets - y_predicted
  x_abs = tf.math.abs(x) #tf.sqrt(x * x)
  is_negative_branch = tf.cast(tf.less(x_abs, 1.0 / sigma_squared), dtype = tf.float32)
  R_negative_branch = 0.5 * x * x * sigma_squared
  R_positive_branch = x_abs - 0.5 / sigma_squared
  losses = is_negative_branch * R_negative_branch + (1.0 - is_negative_branch) * R_positive_branch

  # TODO:
  # Not clear which of these methods of normalization are ideal, or whether it
  # even matters
  #N = tf.reduce_sum(y_mask) / 4.0 + K.epsilon()                      # N = number of positive boxes
  N = tf.cast(tf.shape(y_true)[1], dtype = tf.float32) + K.epsilon() # N = number of proposals
  #N = tf.reduce_sum(y_mask) + K.epsilon()                             # N = number of parameters (i.e., number of positive boxes * 4)
  relevant_loss_terms = y_mask * losses
  return scale_factor * K.sum(relevant_loss_terms) / N
  
def classifier_regression_loss_np(y_true, y_predicted):
  scale_factor = 1.0
  sigma = 1.0
  sigma_squared = sigma * sigma

  # We want to unpack the regression targets and the mask of valid targets into
  # tensors each of the same shape as the predicted: 
  #   (batch_size, num_proposals, 4*(num_classes-1))
  # y_true has shape:
  #   (batch_size, num_proposals, 2, 4*(num_classes-1))
  y_mask = y_true[:,:,0,:]
  y_true_targets = y_true[:,:,1,:]

  # Compute element-wise loss using robust L1 function for all 4 regression
  # targets
  x = y_true_targets - y_predicted
  x_abs = np.sqrt(x * x)
  is_negative_branch = np.less(x_abs, 1.0 / sigma_squared).astype(np.float)
  R_negative_branch = 0.5 * x * x * sigma_squared
  R_positive_branch = x_abs - 0.5 / sigma_squared
  losses = is_negative_branch * R_negative_branch + (1.0 - is_negative_branch) * R_positive_branch

  # TODO:
  # Not clear which of these methods of normalization are ideal, or whether it
  # even matters
  #N = np.sum(y_mask) / 4.0 + 1e-9                      # N = number of positive boxes
  N = float(y_true.shape[1]) + 1e-9                     # N = number of proposals
  #N = np.sum(y_mask) + 1e-9                             # N = number of parameters (i.e., number of positive boxes * 4)
  relevant_loss_terms = y_mask * losses
  return scale_factor * np.sum(relevant_loss_terms) / N
  return scale_factor * np.sum(relevant_loss_terms)
