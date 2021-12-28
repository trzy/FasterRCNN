import numpy as np
import random
from tqdm import tqdm
from tensorflow.keras.optimizers import SGD

from .statistics import TrainingStatistics
from .datasets import voc
from .models import faster_rcnn

def _sample_rpn_minibatch(rpn_map, object_indices, background_indices, rpn_minibatch_size):
  """
  Selects anchors for training and produces a copy of the RPN ground truth
  map with only those anchors marked as trainable.

  Parameters
  ----------
  rpn_map : np.ndarray
    RPN ground truth map of shape
    (batch_size, height, width, num_anchors, 6).
  object_indices : List[np.ndarray]
    For each image in the batch, a map of shape (N, 3) of indices (y, x, k)
    of all N object anchors in the RPN ground truth map.
  background_indices : List[np.ndarray]
    For each image in the batch, a map of shape (M, 3) of indices of all M
    background anchors in the RPN ground truth map.

  Returns
  -------
  np.ndarray
    A copy of the RPN ground truth map with index 0 of the last dimension
    recomputed to include only anchors in the minibatch.
  """
  assert rpn_map.shape[0] == 1, "Batch size must be 1"
  assert len(object_indices) == 1, "Batch size must be 1"
  assert len(background_indices) == 1, "Batch size must be 1"
  positive_anchors = object_indices[0]
  negative_anchors = background_indices[0]
  assert len(positive_anchors) + len(negative_anchors) >= rpn_minibatch_size, "Image has insufficient anchors for RPN minibatch size of %d" % rpn_minibatch_size
  assert len(positive_anchors) > 0, "Image does not have any positive anchors"
  assert rpn_minibatch_size % 2 == 0, "RPN minibatch size must be evenly divisible"

  # Sample, producing indices into the index maps
  num_positive_anchors = len(positive_anchors)
  num_negative_anchors = len(negative_anchors)
  num_positive_samples = min(rpn_minibatch_size // 2, num_positive_anchors) # up to half the samples should be positive, if possible
  num_negative_samples = rpn_minibatch_size - num_positive_samples          # the rest should be negative
  positive_anchor_idxs = random.sample(range(num_positive_anchors), num_positive_samples)
  negative_anchor_idxs = random.sample(range(num_negative_anchors), num_negative_samples)
  
  # Construct index expressions into RPN map
  positive_anchors = positive_anchors[positive_anchor_idxs]
  negative_anchors = negative_anchors[negative_anchor_idxs]
  trainable_anchors = np.concatenate([ positive_anchors, negative_anchors ])
  batch_idxs = np.zeros(len(trainable_anchors), dtype = int)
  trainable_idxs = (batch_idxs, trainable_anchors[:,0], trainable_anchors[:,1], trainable_anchors[:,2], 0)

  # Create a copy of the RPN map with samples set as trainable
  rpn_minibatch_map = rpn_map.copy()
  rpn_minibatch_map[:,:,:,:,0] = 0
  rpn_minibatch_map[trainable_idxs] = 1

  return rpn_minibatch_map

if __name__ == "__main__":

  training_data = voc.Dataset(dir = "../../VOCdevkit/VOC2007", split = "trainval", augment = True, shuffle = True, cache = True)

  model = faster_rcnn.faster_rcnn_model(mode = "train")
  model.compile()
  optimizer = SGD(learning_rate = 1e-3, momentum = 0.9)
  model.compile(optimizer = optimizer, loss = [ None ] * len(model.outputs))

  # Test maps
  height = 256
  width = 256
  anchor_map = np.zeros((height // 16, width // 16, 9 * 4))
  anchor_valid_map = np.zeros((height // 16, width // 16, 9))
  gt_rpn_map = np.zeros((height // 16, width // 16, 9, 6))
  anchor_valid_map[0,0,1] = 1
  anchor_valid_map[1,1,0] = 1
  anchor_map = np.expand_dims(anchor_map, axis = 0)
  anchor_valid_map = np.expand_dims(anchor_valid_map, axis = 0)
  gt_rpn_map = np.expand_dims(gt_rpn_map, axis = 0)
  image_map = np.zeros((1, height, width, 3))
  image_shape_map = np.array([ [ height, width, 3 ] ])  # (1,3)

  y = model.predict(x = [ image_map, image_shape_map, anchor_map, anchor_valid_map, gt_rpn_map ])
  #for i in range(len(y)):
  #  print(y[i].shape)

  # Train loop
  num_epochs = 2
  for epoch in range(1, 1 + num_epochs):
    stats = TrainingStatistics()
    progbar = tqdm(iterable = iter(training_data), total = training_data.num_samples, postfix = stats.get_progbar_postfix())
    for sample in progbar:
      image_data = np.expand_dims(sample.image_data, axis = 0)
      image_shape_map = np.array([ [ image_data.shape[1], image_data.shape[2], image_data.shape[3] ] ])
      anchor_map = np.expand_dims(sample.anchor_map, axis = 0)
      anchor_valid_map = np.expand_dims(sample.anchor_valid_map, axis = 0)
      gt_rpn_map = np.expand_dims(sample.gt_rpn_map, axis = 0)
      gt_rpn_object_indices = [ sample.gt_rpn_object_indices ]
      gt_rpn_background_indices = [ sample.gt_rpn_background_indices ]
      gt_rpn_minibatch_map = _sample_rpn_minibatch(
        rpn_map = gt_rpn_map,
        object_indices = gt_rpn_object_indices,
        background_indices = gt_rpn_background_indices,
        rpn_minibatch_size = 256
      )
      gt_boxes = [ sample.gt_boxes ]
      x = [ image_data, image_shape_map, anchor_map, anchor_valid_map, gt_rpn_minibatch_map ]
      losses = model.train_on_batch(x = x, y = gt_rpn_map, return_dict = True)
      stats.on_training_step(loss = losses)
      progbar.set_postfix(stats.get_progbar_postfix())
      #_, _, _, rpn_class_loss, rpn_regression_loss = model.predict(x = [ image_data, image_shape_map, anchor_map, anchor_valid_map, gt_rpn_map ])
      #print(rpn_class_loss, rpn_regression_loss)
