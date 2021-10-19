#TODO: need a way to separately mark the classifier truth map with POSITIVE regression targets AND positive+negative ones
#Perhaps use -1 for negative or 0 for negative (consistent with elsewhere in region_proposal_network.py)

# 0.4836

# Test detector loss. simple_frcnn formatting is the same as for RPN but our ground truth map is different


import numpy as np
from .models.losses import rpn_class_loss_np, rpn_regression_loss_np, classifier_regression_loss_np

def pytorch_unpack_data(data):
  # Extract and return the pytorch data maps
  pred_loc = data[:,0:4]
  gt_loc=data[:,4:8]
  gt_label=np.squeeze(data[:,8])
  return pred_loc, gt_loc, gt_label

def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = np.abs(diff)
    flag = (abs_diff < (1. / sigma2))
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return np.sum(y)

def pytorch_classifier_regression_loss(pred_loc, gt_loc, gt_label):
  in_weight = np.zeros(gt_loc.shape)
  # Localization loss is calculated only for positive rois.
  # NOTE:  unlike origin implementation, 
  # we don't need inside_weight and outside_weight, they can calculate by gt_label
  #in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1
  pos_labels = (gt_label > 0).reshape((-1, 1))
  pos_mask = np.repeat(pos_labels, repeats=gt_loc.shape[1], axis=1)
  in_weight[pos_mask] = 1

  loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight, sigma = 1.0)
  # Normalize by total number of negtive and positive rois.
  #loc_loss /= ((gt_label >= 0).sum().float()) # ignore gt_label==-1 for rpn_loss
  loc_loss /= np.sum(gt_label >= 0).astype(np.float)
  return loc_loss
  




with open("../simple-faster-rcnn-pytorch/predictions_detector.bin", "rb") as fp:
  data = np.load(fp)  # N x 9, where each row is: predicted locations (4), GT locations (4), included (1) (when >= 0)

pytorch_pred_loc, pytorch_gt_loc, pytorch_gt_label = pytorch_unpack_data(data)
print("pytorch=",pytorch_classifier_regression_loss(pred_loc = pytorch_pred_loc, gt_loc = pytorch_gt_loc, gt_label = pytorch_gt_label))



N = data.shape[0]               # N = number of RoIs (proposals)
num_classes = 20               # doesn't really matter because we will stuff everything into some particular class index
y_true = np.zeros((1,N,2,4*(num_classes-1)))  # batch,num_proposals,2,4*(num_classes-1). 
y_predicted = np.zeros((1,N,4*(num_classes-1))) # batch,num_proposals,4*(num_classes-1)
for i in range(N):
  y_true[0,i,1,1*4+0:1*4+4] = data[i,4:8]   # GT regression targets into class=1 (1*4:1*4+4) slot
  if data[i,8] > 0:                         # positive label
    y_true[0,i,0,1*4+0:1*4+4] = (1, 1, 1, 1)  # set mask indicating these are the valid regression targets here
  y_predicted[0,i,1*4+0:1*4+4] = data[i,0:4]  # predicted targets
num_valid = np.sum(pytorch_gt_label >= 0).astype(np.float)
print("loss=", classifier_regression_loss_np(y_true = y_true, y_predicted = y_predicted)) 

