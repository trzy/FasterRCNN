import numpy as np
from PIL import Image, ImageDraw, ImageFont

def _draw_rectangle(ctx, corners, color, thickness = 4):
  y_min, x_min, y_max, x_max = corners
  ctx.rectangle(xy = [(x_min, y_min), (x_max, y_max)], outline = color, width = thickness)

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
