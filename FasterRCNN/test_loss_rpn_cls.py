import argparse
import numpy as np
from tensorflow.keras import backend as K
from .models.losses import rpn_class_loss, rpn_regression_loss, rpn_class_loss_np, rpn_regression_loss_np


def load_keras_data(gt_filepath, predictions_filepath):
  with open(gt_filepath, "rb") as fp:
    y_true = np.load(fp)
  with open(predictions_filepath, "rb") as fp:
    y_predicted = np.load(fp)
  return y_true, y_predicted

def load_pytorch_data(filepath):
  with open("../simple-rpn/predictions_rpn_cls.bin", "rb") as fp:
    data = np.load(fp)  # N x 2, where column 0 is GT label (-1=ignore, 0=negative sample, 1=object) and column 1 is score (after sigmoid activation)

  N = data.shape[0]
  y_true = np.zeros((1,1,N,9,8))  # batch,height,width,k,8. Use N as width and keep other dimensions 1
  y_predicted = np.zeros((1,1,N,9)) # batch,height,width,k
  for i in range(N):
    y_true[0,0,i,0,0] = 1.0 if (data[i,0] >= 0) else 0.0  # anchor is valid <-- positive AND negative samples 
    y_true[0,0,i,0,1] = 1.0 if (data[i,0] > 0) else 0.0  # object (positive samples only)
    y_true[0,0,i,0,2] = 1.0 if (data[i,0] > 0) else -1.0 # class (just use 1.0) or negative if not object
    y_predicted[0,0,i,0] = data[i,1]

  return y_true, y_predicted

def evaluate_keras_loss(y_true, y_predicted):
  print("cls loss=", rpn_class_loss_np(y_true = y_true, y_predicted = y_predicted)) 
  return K.eval(rpn_class_loss(y_true = K.variable(y_true), y_predicted = K.variable(y_predicted)))

def evaluate_pytorch_loss(y_true, y_predicted):
  # Convert to PyTorch-style tensors

  N = y_true.shape[0] * y_true.shape[1] * y_true.shape[2] * y_true.shape[3] # batch_size*y*x*k
  rpn_scores = np.zeros((N, 1))
  gt_rpn_label = np.zeros(N)
  i = 0
  for batch in range(y_true.shape[0]):
    for y in range(y_true.shape[1]):
      for x in range(y_true.shape[2]):
        for k in range(y_true.shape[3]):
          rpn_scores[i,0] = y_predicted[batch,y,x,k]
          gt_rpn_label[i] = y_true[batch,y,x,k,1]
          if y_true[batch,y,x,k,0] < 1: # excluded
            gt_rpn_label[i] = -1        # mark as excluded
          i += 1

  # Evaluate
  return _pytorch_cls_loss(rpn_scores, gt_rpn_label)

def _pytorch_cls_loss(rpn_scores, gt_rpn_label):
  rpn_scores = np.squeeze(rpn_scores[gt_rpn_label > -1])  # filter out neutral samples and convert from [N,1] -> [N]
  _gt_rpn_label = gt_rpn_label[gt_rpn_label > -1]
  eps = 1e-7
  rpn_cls_loss = np.sum(-(_gt_rpn_label * np.log(rpn_scores + eps) + (1.0 - _gt_rpn_label) * np.log(1.0 - rpn_scores + eps))) / len(_gt_rpn_label)
#F.binary_cross_entropy(rpn_scores, _gt_rpn_label)
  return rpn_cls_loss

parser = argparse.ArgumentParser("test_loss_rpn_cls")
group = parser.add_argument_group("Operation")
group_ex = parser.add_mutually_exclusive_group()
group_ex.add_argument("--load-pytorch", metavar = "file", type = str, action = "store", help = "Load PyTorch inputs from single file")
group_ex.add_argument("--load-keras", action = "store_true", help = "Load Keras inputs from keras_rpn_predictions_cls.bin and keras_rpn_gt_cls.bin")
options = parser.parse_args()

if options.load_pytorch:
  y_true, y_predicted = load_pytorch_data(options.load_pytorch)
else:
  y_true, y_predicted = load_keras_data(gt_filepath = "keras_rpn_gt_cls.bin", predictions_filepath = "keras_rpn_predictions_cls.bin")

keras_loss = evaluate_keras_loss(y_true, y_predicted)
pytorch_loss = evaluate_pytorch_loss(y_true, y_predicted)
print("Keras:   %f" % keras_loss)
print("PyTorch: %f" % pytorch_loss)

