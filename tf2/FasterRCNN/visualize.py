#
# Faster R-CNN in PyTorch and TensorFlow 2 w/ Keras
# tf2/FasterRCNN/visualize.py
# Copyright 2021-2022 Bart Trzynadlowski
#
# Routines for visualizing model results and debug information.
#

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor

def _draw_rectangle(ctx, corners, color, thickness = 4):
  y_min, x_min, y_max, x_max = corners
  ctx.rectangle(xy = [(x_min, y_min), (x_max, y_max)], outline = color, width = thickness)

def _draw_text(image, text, position, color, scale = 1.0, offset_lines = 0):
  """
  Parameters
  ----------
  image : PIL.Image
    Image object to draw on.
  text : str
    Text to render.
  position : Tuple[float, float]
    Location of top-left corner of text string in pixels.
  offset_lines : float
    Number of lines to offset the vertical position by, where a line is the
    text height.
  """
  font = ImageFont.load_default()
  text_size = font.getsize(text)
  text_image = Image.new(mode = "RGBA", size = text_size, color = (0, 0, 0, 0))
  ctx = ImageDraw.Draw(text_image)
  ctx.text(xy = (0, 0), text = text, font = font, fill = color)
  scaled = text_image.resize((round(text_image.width * scale), round(text_image.height * scale)))
  position = (round(position[0]), round(position[1] + offset_lines * scaled.height))
  image.paste(im = scaled, box = position, mask = scaled)

def _class_to_color(class_index):
  return list(ImageColor.colormap.values())[class_index + 1]

def show_anchors(output_path, image, anchor_map, anchor_valid_map, gt_rpn_map, gt_boxes, display = False):
  ctx = ImageDraw.Draw(image, mode = "RGBA")
  
  # Draw all ground truth boxes with thick green lines
  for box in gt_boxes:
    _draw_rectangle(ctx, corners = box.corners, color = (0, 255, 0))

  # Draw all object anchor boxes in yellow
  for y in range(anchor_valid_map.shape[0]):
    for x in range(anchor_valid_map.shape[1]):
      for k in range(anchor_valid_map.shape[2]):  
        if anchor_valid_map[y,x,k] <= 0 or gt_rpn_map[y,x,k,0] <= 0:
          continue  # skip anchors excluded from training
        if gt_rpn_map[y,x,k,1] < 1:
          continue  # skip background anchors
        height = anchor_map[y,x,k*4+2]
        width = anchor_map[y,x,k*4+3]
        cy = anchor_map[y,x,k*4+0]
        cx = anchor_map[y,x,k*4+1]
        corners = (cy - 0.5 * height, cx - 0.5 * width, cy + 0.5 * height, cx + 0.5 * width)
        _draw_rectangle(ctx, corners = corners, color = (255, 255, 0), thickness = 3)
 
  image.save(output_path)
  if display:
    image.show()

def show_detections(output_path, show_image, image, scored_boxes_by_class_index, class_index_to_name):
  # Draw all results
  ctx = ImageDraw.Draw(image, mode = "RGBA")
  color_idx = 0
  for class_index, scored_boxes in scored_boxes_by_class_index.items():
    for i in range(scored_boxes.shape[0]):
      scored_box = scored_boxes[i,:]
      class_name = class_index_to_name[class_index]
      text = "%s %1.2f" % (class_name, scored_box[4])
      color = _class_to_color(class_index = class_index)
      _draw_rectangle(ctx = ctx, corners = scored_box[0:4], color = color, thickness = 2)
      _draw_text(image = image, text = text, position = (scored_box[1], scored_box[0]), color = color, scale = 1.5, offset_lines = -1)

  # Output
  if show_image:
    image.show()
  if output_path is not None:
    image.save(output_path)
    print("Wrote detection results to '%s'" % output_path)

