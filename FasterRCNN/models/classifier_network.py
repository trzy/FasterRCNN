from . import region_proposal_network
from .roi_pooling_layer import RoIPoolingLayer

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import models

def layers(input_map, proposal_boxes):
  """
  Constructs classifier network from conv. net output map and proposal boxes
  scaled to the map's coordinate space. Shape of proposal_boxes is (N,4) where
  N is the number of proposals and the second dimension contains:

    0: y_min
    1: x_min
    2: y_max
    3: x_max
  """
  assert len(input_map.shape) == 4

  pool = RoIPoolingLayer(pool_size = 7)([input_map, proposal_boxes])
  return pool




