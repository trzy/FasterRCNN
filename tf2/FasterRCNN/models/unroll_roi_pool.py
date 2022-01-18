#
# Faster R-CNN in PyTorch and TensorFlow 2 w/ Keras
# tf2/FasterRCNN/models/unroll_roi_pool.py
# Copyright 2021-2022 Bart Trzynadlowski
#
# Script to generate unrolled pooling functions for use in RoIPoolingLayer. The
# default, generic implementation capable of handling any pool dimensions uses
# multiple nested functions applied with tf.map_fn(), which is extremely slow.
# Unrolling along the pool width and height speeds things up.
#

import argparse


if __name__ == "__main__":
  parser = argparse.ArgumentParser("unroll_roi_pool")
  parser.add_argument("--pool-size", metavar = "size", type = int, action = "store", default = "7", help = "Pool width and height")
  parser.add_argument("--channels", metavar = "size", type = int, action = "store", default = "512", help = "Number of channels (in output of conv. net)")
  options = parser.parse_args()

  assert options.pool_size > 0

  pool_width = options.pool_size
  pool_height = options.pool_size
  num_channels = options.channels

  print("  @tf.function")
  print("  def _compute_pooled_rois_%dx%dx%d(feature_map, rois):" % (pool_height, pool_width, num_channels))
  print("    # Special case: %dx%dx%d, unrolled pool width and height (%dx%d=%d)" % (pool_height, pool_width, num_channels, pool_height, pool_width, pool_height * pool_width))
  print("    return tf.map_fn(")
  print("      fn = lambda roi: tf.reshape(")
  print("        tf.stack([")
  for y in range(pool_height):
    for x in range(pool_width):
      print("          # y=%d,x=%d" % (y, x))
      print("          tf.math.reduce_max(")
      print("            tf.slice(")
      print("              feature_map[ roi[0]:roi[0]+roi[2], roi[1]:roi[1]+roi[3], 0:%d ]," % num_channels)
      print("              [")
      print("                tf.cast(%d * (tf.cast(roi[2], dtype = tf.float32) / %d), dtype = tf.int32)," % (y, pool_height))
      print("                tf.cast(%d * (tf.cast(roi[3], dtype = tf.float32) / %d), dtype = tf.int32)," % (x, pool_width))
      print("                0")
      print("              ],")
      print("              [")
      if (y + 1) < pool_height:
        print("                tf.math.maximum(1, tf.cast((%d + 1) * (tf.cast(roi[2], dtype = tf.float32) / %d), dtype = tf.int32) - tf.cast(%d * (tf.cast(roi[2], dtype = tf.float32) / %d), dtype = tf.int32))," % (y, pool_height, y, pool_height))
      else:
        print("                tf.math.maximum(1, roi[2] - tf.cast(%d * (tf.cast(roi[2], dtype = tf.float32) / %d), dtype = tf.int32))," % (y, pool_height))
      if (x + 1) < pool_width:
        print("                tf.math.maximum(1, tf.cast((%d + 1) * (tf.cast(roi[3], dtype = tf.float32) / %d), dtype = tf.int32) - tf.cast(%d * (tf.cast(roi[3], dtype = tf.float32) / %d), dtype = tf.int32))," % (x, pool_width, x, pool_width))
      else:
        print("                tf.math.maximum(1, roi[3] - tf.cast(%d * (tf.cast(roi[3], dtype = tf.float32) / %d), dtype = tf.int32))," % (x, pool_width))
      print("                %d" % num_channels)
      print("              ]")
      print("            ),")
      print("            axis = (1,0)")
      print("          ),")
  print("        ]),")
  print("        shape = (%d,%d,%d)" % (pool_height, pool_width, num_channels))
  print("      ),")
  print("      elems = rois,")
  print("      fn_output_signature = tf.float32")
  print("    )")


# Original code:
# for y in range(pool_height):
#   for x in range(pool_width):
#     print("          # y=%d,x=%d" % (y, x))
#     print("          tf.math.reduce_max(")
#     print("            tf.slice(")
#     print("              feature_map[ roi[0]:roi[0]+roi[2], roi[1]:roi[1]+roi[3], 0:%d ]," % num_channels)
#     print("              [")
#     print("                tf.cast(%d * (tf.cast(roi[2], dtype = tf.float32) / tf.cast(%d, dtype = tf.float32)), dtype = tf.int32)," % (y, pool_height))
#     print("                tf.cast(%d * (tf.cast(roi[3], dtype = tf.float32) / tf.cast(%d, dtype = tf.float32)), dtype = tf.int32)," % (x, pool_width))
#     print("                0")
#     print("              ],")
#     print("              [")
#     print("                tf.math.maximum(tf.cond((tf.cast(%d, dtype = tf.int32) + 1) < %d, lambda: tf.cast((%d + 1) * (tf.cast(roi[2], dtype = tf.float32) / tf.cast(%d, dtype = tf.float32)), dtype = tf.int32), lambda: roi[2]) - tf.cast(%d * (tf.cast(roi[2], dtype = tf.float32) / tf.cast(%d, dtype = tf.float32)), dtype = tf.int32), 1)," % (y, pool_height, y, pool_height, y, pool_height))
#     print("                tf.math.maximum(tf.cond((tf.cast(%d, dtype = tf.int32) + 1) < %d, lambda: tf.cast((%d + 1) * (tf.cast(roi[3], dtype = tf.float32) / tf.cast(%d, dtype = tf.float32)), dtype = tf.int32), lambda: roi[3]) - tf.cast(%d * (tf.cast(roi[3], dtype = tf.float32) / tf.cast(%d, dtype = tf.float32)), dtype = tf.int32), 1)," % (x, pool_width, x, pool_width, x, pool_width))
#     print("                %d" % num_channels)
#     print("              ]")
#     print("            ),")
#     print("            axis = (1,0)")
#     print("          ),")
# print("        ]),")
# print("        shape = (%d,%d,%d)" % (pool_height, pool_width, num_channels))
# print("      ),")
# print("      elems = rois,")
# print("      fn_output_signature = tf.float32")
# print("    )")
