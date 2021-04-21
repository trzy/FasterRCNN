from ..models.rpn_loss import rpn_regression_loss_np

import numpy as np

if __name__ == "__main__":
  # Creates an artificial prediction and ground truth pair to test the box
  # regression loss function. All anchors are marked as valid and a single
  # anchor is marked as positive. Only it should be reflected in the loss.

  predicted_t = np.array([10, 20, 30, 40])
  true_t      = np.array([11, 22, 33, 44])

  y_predicted = np.zeros((1, 2, 2, 9*4))
  y_true      = np.zeros((1, 2, 2, 9, 8))

  y_predicted[0,0,0,0:9*4] = -50 * np.ones(9*4)   # all anchors at (0,0) are invalid
  y_predicted[0,1,0,0:9*4] = -60 * np.ones(9*4)   # ""
  y_predicted[0,1,1,0:9*4] = -100 * np.ones(9*4)  # ""
  y_predicted[0,0,1,0:9*4] = -1 * np.ones(9*4)    # anchors #0, #2-#8 at position (0,1) are invalid and all regression values set to -1
  y_predicted[0,0,1,4:8] = predicted_t            # anchor #1 is valid

  # Make all anchors valid (but not positive)
  y_true[0,:,:,:,0] = 1.0

  # Anchor #1 at (0,1)
  y_true[0,0,1,1,0] = 1.0         # is valid
  y_true[0,0,1,1,1] = 1.0         # is positive example
  y_true[0,0,1,1,2] = 1.0 + 2.0   # box number 2
  y_true[0,0,1,1,3] = 0.85        # IoU
  y_true[0,0,1,1,4] = true_t[0]   # ty
  y_true[0,0,1,1,5] = true_t[1]   # tx
  y_true[0,0,1,1,6] = true_t[2]   # th
  y_true[0,0,1,1,7] = true_t[3]   # tw

  loss = rpn_regression_loss_np(y_true, y_predicted)

  print("Computed Loss =", loss)

  # Compute by hand for the one valid (positive && valid) anchor that we have
  x = true_t - predicted_t
  x_abs = np.abs(x)
  sigma = 3.0
  is_negative_branch = np.less(x_abs, 1.0).astype(np.float)
  R_negative_branch = 0.5 * sigma * sigma * x * x
  R_positive_branch = x_abs - 0.5 / (sigma * sigma)
  loss_all_components = is_negative_branch * R_negative_branch + (1.0 - is_negative_branch) * R_positive_branch
  N_cls = 2 * 2 * 9 # all anchors included in computation
  loss_expected = np.sum(loss_all_components) / N_cls

  print("Expected Loss = ", loss_expected)

  error = loss / loss_expected - 1.0
  print("Error = %1.1f%%" % (100.0 * error))
  assert error < 0.001, "** Test FAILED **"
  print("** Test PASSED **")
