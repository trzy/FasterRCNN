from .dataset import VOC
from .models import region_proposal_network
from .models import intersection_over_union
from .models.nms import nms

import imageio
import itertools
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random

def draw_rectangle(ctx, x_min, y_min, x_max, y_max, color, thickness = 4):
  ctx.rectangle(xy = [(x_min, y_min), (x_max, y_max)], outline = color, width = thickness)

def draw_filled_rectangle(ctx, x_min, y_min, x_max, y_max, color):
  ctx.rectangle(xy = [(x_min, y_min), (x_max, y_max)], fill = color)

def show_annotated_image(voc, filename, draw_anchor_points = True, draw_anchor_intersections = False, image_input_map = None, anchor_map = None):
  # Load image (automatically re-scaled)
  filepath = voc.get_full_path(filename = filename)
  info = voc.get_image_info(path = filepath)
  image = info.load_image()
  boxes = info.get_boxes()

  # Draw ground truth boxes
  _draw_ground_truth_boxes(image = image, boxes = boxes)

  # Draw anchor boxes and intersection areas, if requested
  if draw_anchor_intersections or draw_anchor_points:
    _draw_anchor_box_intersections(image = image, ground_truth_boxes = boxes, draw_anchor_points = draw_anchor_points, input_image_shape = image_input_map.shape[1:])

  # Display image
  image.show()
  image.save("out_gt.png")

def draw_text(image, text, position, color, scale = 1.0, offset_lines = 0):
  """
  position:     Location of top-left corner of string in pixels.
  offset_lines: Number of lines (a line being the text height in pixels) to
                offset the y position by.
  """
  font = ImageFont.load_default()
  text_size = font.getsize(text)
  text_image = Image.new(mode = "RGBA", size = text_size, color = (0, 0, 0, 0))
  ctx = ImageDraw.Draw(text_image)
  ctx.text(xy = (0, 0), text = text, font = font, fill = color)
  scaled = text_image.resize((round(text_image.width * scale), round(text_image.height * scale)))
  position = (round(position[0]), round(position[1] + offset_lines * scaled.height))
  image.paste(im = scaled, box = position, mask = scaled)

def show_objects(image, boxes_by_class_name):
  # Create a selection of colors
  colors = [
    (0, 255, 0),    # green
    (0, 255, 255),  # yellow
    (0, 0, 255),    # red
    (255, 0, 0),    # blue
    (255, 255, 0),  # cyan
    (255, 0, 255),  # purple
    (0, 128, 255),  # orange
    (128, 255, 0),
    (128, 0, 255),
    (0, 255, 128),
    (255, 0, 128)
  ]

  # Draw all results
  ctx = ImageDraw.Draw(image, mode = "RGBA")
  color_idx = 0
  for class_name, boxes in boxes_by_class_name.items():
    for box in boxes:
      color = colors[color_idx]
      color_idx = (color_idx + 1) % len(colors)
      draw_rectangle(ctx = ctx, x_min = box[1], y_min = box[0], x_max = box[3], y_max = box[2], color = color, thickness = 2)
      draw_text(image = image, text = class_name, position = (box[1], box[0]), color = color, scale = 2.5, offset_lines = -1)
  image.show()

def show_proposed_regions(voc, filename, y_true, y_class, y_regression):
  # Load image and scale appropriately
  filepath = voc.get_full_path(filename = filename)
  info = voc.get_image_info(path = filepath)
  image = info.load_image()  
  ctx = ImageDraw.Draw(image, mode = "RGBA")

  # Get all anchors for this image size
  anchor_boxes, anchor_boxes_valid = region_proposal_network.compute_all_anchor_boxes(input_image_shape = (info.height, info.width, 3))

  # Draw positive anchors we got correct (true positives) as green and those we mispredicted as orange (false negatives)
  """
  for y in range(y_class.shape[1]):
    for x in range(y_class.shape[2]):
      for k in range(y_class.shape[3]):
        anchor_box = anchor_boxes[y,x,k*4+0:k*4+4]
        if y_class[0,y,x,k] > 0.5 and y_true[0,y,x,k,1] > 0.5:    # true positive
          draw_filled_rectangle(ctx = ctx, x_min = anchor_box[1] - 0.5 * anchor_box[3], x_max = anchor_box[1] + 0.5 * anchor_box[3], y_min = anchor_box[0] - 0.5 * anchor_box[2], y_max = anchor_box[0] + 0.5 * anchor_box[2], color = (0, 255, 0, 64))
        elif y_class[0,y,x,k] < 0.5 and y_true[0,y,x,k,1] > 0.5:  # false negative
          draw_filled_rectangle(ctx = ctx, x_min = anchor_box[1] - 0.5 * anchor_box[3], x_max = anchor_box[1] + 0.5 * anchor_box[3], y_min = anchor_box[0] - 0.5 * anchor_box[2], y_max = anchor_box[0] + 0.5 * anchor_box[2], color = (255, 100, 0, 64))
  """
  # Extract proposals (which also performs NMS)
  final_proposals = region_proposal_network.extract_proposals(y_predicted_class = y_class, y_predicted_regression = y_regression, input_image_shape = info.shape(), anchor_boxes = anchor_boxes, anchor_boxes_valid = np.expand_dims(anchor_boxes_valid, axis=0))

  # Label proposals, so we can see whether any of them would actually be assigned to a ground truth box
  final_proposals, y_class_labels, _ = region_proposal_network.label_proposals(proposals = final_proposals, ground_truth_object_boxes = info.get_boxes(), num_classes = voc.num_classes, min_iou_threshold = 0.0, max_iou_threshold = 0.5)

  # Draw boxes
  for i in range(final_proposals.shape[0]):
    # Color is green if this proposal would be assigned to a ground truth box, else yellow
    if np.argmax(y_class_labels[i]) != 0:
      color = (0, 255, 0, 255)
    else:
      continue
      color = (255, 255, 0, 255)
    y_min, x_min, y_max, x_max = final_proposals[i,0:4]
    #print("proposal =", y_min, x_min, y_max, x_max)
    draw_rectangle(ctx = ctx, x_min = x_min, y_min = y_min, x_max = x_max, y_max = y_max, color = color, thickness = 1)

  # Write out image
  image.show()
  image.save("out.png")

  # Write out each proposal with its class name
  #y_proposal_classes = region_proposal_network.label_proposals(proposals = final_proposals, ground_truth_object_boxes = info.get_boxes(), num_classes = voc.num_classes) 
  #for i in range(final_proposals.shape[0]):
  #  y1, x1, y2, x2 = final_proposals[i,0:4]
  #  proposal_image = image.crop(box = (x1, y1, x2, y2))
  #  class_idx = np.argmax(y_proposal_classes[i,:])
  #  proposal_image.save("out_%d_%s.png" % (i, voc.index_to_class_name[class_idx]))

def _draw_ground_truth_boxes(image, boxes):
  # Draw green boxes
  ctx = ImageDraw.Draw(image)
  for box in boxes:
    draw_rectangle(ctx, x_min = box.corners[1], y_min = box.corners[0], x_max = box.corners[3], y_max = box.corners[2], color = (0, 255, 0, 255))
    print("box=%s" % str(box))

def _debug_anchors(anchor_boxes, ground_truth_boxes):

  for box in ground_truth_boxes:
    cx = 0.5 * (box.corners[1] + box.corners[3])
    cy = 0.5 * (box.corners[0] + box.corners[2])
    print("Box: class=%d, center=(%1.0f,%1.0f)" % (box.class_index, cx, cy))

    y1, x1, y2, x2 = box.corners
    
    intersecting_anchors = []

    for y in range(anchor_boxes.shape[0]):
      for x in range(anchor_boxes.shape[1]):
        for k in range(anchor_boxes.shape[2] // 4):
          ax1 = anchor_boxes[y,x,k*4+1] - 0.5 * anchor_boxes[y,x,k*4+3]
          ax2 = anchor_boxes[y,x,k*4+1] + 0.5 * anchor_boxes[y,x,k*4+3]
          ay1 = anchor_boxes[y,x,k*4+0] - 0.5 * anchor_boxes[y,x,k*4+2]
          ay2 = anchor_boxes[y,x,k*4+0] + 0.5 * anchor_boxes[y,x,k*4+2]
          no_intersection = ax2 < x1 or ax1 > x2 or ay2 < y1 or ay1 > y2
          intersects = not no_intersection
          if intersects:
            iou = intersection_over_union.intersection_over_union(box1 = box.corners, box2 = (ay1, ax1, ay2, ax2))
            intersecting_anchors.append((y, x, k, iou))

    sorted_anchors = sorted(intersecting_anchors, key = lambda a: a[3], reverse = True)
    print("  Num anchors=", len(sorted_anchors))
    for a in sorted_anchors:
      y, x, k, iou = a
      ax1 = anchor_boxes[y,x,k*4+1] - 0.5 * anchor_boxes[y,x,k*4+3]
      ax2 = anchor_boxes[y,x,k*4+1] + 0.5 * anchor_boxes[y,x,k*4+3]
      ay1 = anchor_boxes[y,x,k*4+0] - 0.5 * anchor_boxes[y,x,k*4+2]
      ay2 = anchor_boxes[y,x,k*4+0] + 0.5 * anchor_boxes[y,x,k*4+2]
      print("  Anchor: %d,%d  %1.1f x %1.1f  IoU=%f    (%f,%f,%f,%f)" % (anchor_boxes[y,x,k*4+1], anchor_boxes[y,x,k*4+0], anchor_boxes[y,x,k*4+3], anchor_boxes[y,x,k*4+2], iou, ay1, ax1, ay2, ax2))


def _draw_anchor_box_intersections(image, ground_truth_boxes, draw_anchor_points, input_image_shape):
  ctx = ImageDraw.Draw(image, mode = "RGBA")

  anchor_boxes, anchor_boxes_valid = region_proposal_network.compute_all_anchor_boxes(input_image_shape = input_image_shape)

  ground_truth_map, positive_anchors, negative_anchors = region_proposal_network.compute_ground_truth_map(ground_truth_object_boxes = ground_truth_boxes, anchor_boxes = anchor_boxes, anchor_boxes_valid = anchor_boxes_valid)

  for y in range(anchor_boxes.shape[0]):
    for x in range(anchor_boxes.shape[1]):
      anchors_per_location = int(anchor_boxes.shape[2] / 4)
      assert int(anchor_boxes.shape[2]) % int(4) == 0
      for anchor_num in range(anchors_per_location):
        # Extract box
        box = anchor_boxes[y, x, anchor_num * 4 : anchor_num * 4 + 4]
        center_y = box[0]
        center_x = box[1]
        height = box[2]
        width = box[3]
        anchor_y_min = center_y - 0.5 * height
        anchor_x_min = center_x - 0.5 * width
        anchor_y_max = center_y + 0.5 * height
        anchor_x_max = center_x + 0.5 * width

        # Draw anchor center pos
        if draw_anchor_points:
          size = 1
          any_positive = True in [ ground_truth_map[y,x,k,2] > 0 for k in range(ground_truth_map.shape[2]) ]
          color = (0, 255, 0, 255) if any_positive else (255, 0, 0, 255)
          draw_filled_rectangle(ctx, y_min = center_y - size, x_min = center_x - size, y_max = center_y + size, x_max = center_x + size, color = color)

        ## Does it intersect with any ground truth box?
        #for box in ground_truth_boxes:
        #  box1 = (anchor_y_min, anchor_x_min, anchor_y_max, anchor_x_max)
        #  box2 = (box.y_min, box.x_min, box.y_max, box.x_max)

        #  iou = intersection_over_union.intersection_over_union(box1 = box1, box2 = box2)
        #  if iou > 0.7:
        #    # Render intersection area
        #    intersection_box = intersection_over_union.intersection(box1 = box1, box2 = box2)
        #    draw_filled_rectangle(ctx, y_min = intersection_box[0], x_min = intersection_box[1], y_max = intersection_box[2], x_max = intersection_box[3], color = (255, 0, 0, 64))

        #    # Draw the anchor box outline
        #    draw_rectangle(ctx, x_min = anchor_x_min, y_min = anchor_y_min, x_max = anchor_x_max, y_max = anchor_y_max, color = (255, 0, 0, 255))

  # Draw all anchor boxes labeled as object in yellow
  print("num_positive_anchors=%d" % len(positive_anchors))
  for i in range(len(positive_anchors)):
    y = positive_anchors[i][0]
    x = positive_anchors[i][1]
    k = positive_anchors[i][2]

    print("anchor=[%d,%d,%d] -- %d,%d -- %d x %d" % (y, x, k, anchor_boxes[y,x,k*4+1], anchor_boxes[y,x,k*4+0], anchor_boxes[y,x,k*4+2], anchor_boxes[y,x,k*4+3]))

    # Extract box and draw
    box = anchor_boxes[y, x, k * 4 : k * 4 + 4]
    center_y = box[0]
    center_x = box[1]
    height = box[2]
    width = box[3]
    anchor_y_min = center_y - 0.5 * height
    anchor_x_min = center_x - 0.5 * width
    anchor_y_max = center_y + 0.5 * height
    anchor_x_max = center_x + 0.5 * width

    draw_rectangle(ctx, x_min = anchor_x_min, y_min = anchor_y_min, x_max = anchor_x_max, y_max = anchor_y_max, color = (255, 255, 0, 255))

  _debug_anchors(anchor_boxes = anchor_boxes, ground_truth_boxes = ground_truth_boxes)
