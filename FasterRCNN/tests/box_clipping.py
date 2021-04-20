#
# TODO: Add more tests by creating different RoIs and maybe
# a pool size of 3x3
#

from ..models.region_proposal_network import clip_box_coordinates_to_map_boundaries

import numpy as np

if __name__ == "__main__":
  # Create some boxes in a fictitious 640x480 image
  boxes = np.array([
    # y_min, x_min, y_max, x_max
    [ 100, 200, 400, 300 ], # retained
    [ 0, 0, 480, 640 ],     # clip to: 0, 0, 479, 639
    [ 0, 0, 479, 639 ],     # retained
    [ 1, 1, 479, 639 ],     # retained
    [ 1, 1, 481, 641 ],     # clip to: 1, 1, 479, 639
    [ -10, 10, -1, 20 ],    # removed (completely out of bounds)
    [ -10, 10, 1, 20 ],     # clip to: 0, 10, 1, 20
    [ 10, -10, 20, 30 ],    # clip to: 10, 0, 20, 30
    [ 10, 640, 20, 650 ],   # removed (completely out of bounds)
    [ 480, 10, 490, 20 ],   # removed (completely out of bounds)
    [ -10, -10, 490, 650 ]  # clip to: 0, 0, 479, 639
  ])

  expected_result = np.array([
    [ 100, 200, 400, 300 ], # retained
    [ 0, 0, 479, 639 ],     # clip to: 0, 0, 479, 639
    [ 0, 0, 479, 639 ],     # retained
    [ 1, 1, 479, 639 ],     # retained
    [ 1, 1, 479, 639 ],     # clip to: 1, 1, 479, 639
    [ 0, 10, 1, 20 ],       # clip to: 0, 10, 1, 20
    [ 10, 0, 20, 30 ],      # clip to: 10, 0, 20, 30
    [ 0, 0, 479, 639 ]      # clip to: 0, 0, 479, 639
  ])

  clipped_boxes = clip_box_coordinates_to_map_boundaries(boxes = boxes, map_shape = (480, 640, 3))
  print(clipped_boxes)

  # Check result
  assert np.array_equal(clipped_boxes, expected_result), "** Test FAILED **"
  print("** Test PASSED **")