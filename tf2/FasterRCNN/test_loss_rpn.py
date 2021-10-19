import argparse
import numpy as np
from tensorflow.keras import backend as K
from .models.losses import rpn_class_loss, rpn_regression_loss, rpn_class_loss_np, rpn_regression_loss_np



def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = np.abs(diff)
    flag = (abs_diff < (1. / sigma2))
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return np.sum(y)

def _pytorch_regression_loss(pred_loc, gt_loc, gt_label):
  in_weight = np.zeros(gt_loc.shape)
  # Localization loss is calculated only for positive rois.
  # NOTE:  unlike origin implementation, 
  # we don't need inside_weight and outside_weight, they can calculate by gt_label
  #in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1
  pos_labels = (gt_label > 0).reshape((-1, 1))
  pos_mask = np.repeat(pos_labels, repeats=gt_loc.shape[1], axis=1)
  in_weight[pos_mask] = 1

  loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight, sigma = 3.0)
  # Normalize by total number of negtive and positive rois.
  #loc_loss /= ((gt_label >= 0).sum().float()) # ignore gt_label==-1 for rpn_loss
  loc_loss /= np.sum(gt_label >= 0).astype(np.float)  # gt_label>=0 counts positive and negative labels
  return loc_loss







def load_keras_data(gt_filepath, predictions_filepath):
  with open(gt_filepath, "rb") as fp:
    y_true = np.load(fp)
  with open(predictions_filepath, "rb") as fp:
    y_predicted = np.load(fp)
  return y_true, y_predicted

def load_pytorch_data(filepath):
  with open("../simple-rpn/predictions_rpn.bin", "rb") as fp:
    data = np.load(fp)  # N x 9, where each row is: predicted locations (4), GT locations (4), included (1) (when >= 0)

  N = data.shape[0]
  y_true = np.zeros((1,1,N,9,8))  # batch,height,width,k,8. Use N as width and keep other dimensions 1
  y_predicted = np.zeros((1,1,N,9*4)) # batch,height,width,k*4
  for i in range(N):
    y_true[0,0,i,0,4:8] = data[i,4:8] # GT regression targets into k=0 slot
    y_true[0,0,i,0,0] = 1.0 if (data[i,8] >= 0) else 0.0          # anchor is valid <-- positive AND negative samples 
    y_true[0,0,i,0,1] = 1.0 if (data[i,8] > 0) else 0.0  # object (positive samples only)
    y_true[0,0,i,0,2] = 1.0 if (data[i,8] > 0) else -1.0 # class (just use 1.0) or negative if not object
    y_predicted[0,0,i,0*4+0:0*4+4] = data[i,0:4]  # predicted targets

  return y_true, y_predicted

def evaluate_keras_loss(y_true, y_predicted):
  print("loss=", rpn_regression_loss_np(y_true = y_true, y_predicted = y_predicted)) 
  return K.eval(rpn_regression_loss(y_true = K.variable(y_true), y_predicted = K.variable(y_predicted)))

def evaluate_keras_loss_manually(y_true, y_predicted):
  sum_of_l1 = 0.0
  n = 0
  
  for batch in range(y_true.shape[0]):
    for y in range(y_true.shape[1]):
      for x in range(y_true.shape[2]):
        for k in range(y_true.shape[3]):
          if y_true[batch,y,x,k,0] < 1: # only count valid anchors (those included in minibatch), positive and negative samples
            continue
          n += 1
          if y_true[batch,y,x,k,1] < 1: # loss only computed for positive anchors
            continue

          yt = y_true[batch,y,x,k,4:8]
          yp = y_predicted[batch,y,x,k*4+0:k*4+4]
          sigma = 3.0
          sigma2 = sigma * sigma
          diff = yt - yp
          abs_diff = np.abs(diff)
          out = np.zeros(4)
          for i in range(4):
            if abs_diff[i] < (1 / sigma2):
              out[i] = 0.5 * sigma2 * diff[i] * diff[i]
            else:
              out[i] = abs_diff[i] - 0.5 / sigma2
          l1 = np.sum(out)

          sum_of_l1 += l1

  return sum_of_l1 / n
          

def evaluate_pytorch_loss(y_true, y_predicted):
  # Convert to PyTorch-style tensors
  N = y_true.shape[0] * y_true.shape[1] * y_true.shape[2] * y_true.shape[3] # batch_size*y*x*k
  pred_loc = np.zeros((N, 4))
  gt_loc = np.zeros((N, 4))
  gt_label = np.ones((N,1))
  i = 0
  for batch in range(y_true.shape[0]):
    for y in range(y_true.shape[1]):
      for x in range(y_true.shape[2]):
        for k in range(y_true.shape[3]):
          pred_loc[i,:] = y_predicted[batch,y,x,k*4+0:k*4+4]
          gt_loc[i,:] = y_true[batch,y,x,k,4:8]
          gt_label[i,0] = y_true[batch,y,x,k,1]
          if y_true[batch,y,x,k,0] < 1: # excluded
            gt_label[i,0] = -1          # mark as excluded
          i += 1

  # Evaluate
  return _pytorch_regression_loss(pred_loc, gt_loc, gt_label)
        


   
  return 0.0

parser = argparse.ArgumentParser("test_loss_rpn")
group = parser.add_argument_group("Operation")
group_ex = parser.add_mutually_exclusive_group()
group_ex.add_argument("--load-pytorch", metavar = "file", type = str, action = "store", help = "Load PyTorch inputs from single file")
group_ex.add_argument("--load-keras", action = "store_true", help = "Load Keras inputs from keras_rpn_predictions.bin and keras_rpn_gt.bin")
options = parser.parse_args()

if options.load_pytorch:
  y_true, y_predicted = load_pytorch_data(options.load_pytorch)
else:
  y_true, y_predicted = load_keras_data(gt_filepath = "keras_rpn_gt.bin", predictions_filepath = "keras_rpn_predictions.bin")

keras_loss = evaluate_keras_loss(y_true, y_predicted)
pytorch_loss = evaluate_pytorch_loss(y_true, y_predicted)
print("Keras:    %f" % keras_loss)
print("  Manual: %f" % evaluate_keras_loss_manually(y_true, y_predicted))
print("PyTorch: %f" % pytorch_loss)

